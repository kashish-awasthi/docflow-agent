"""A tool-calling agent: the LLM decides which tools to call to answer a request or carry out a task."""
import json
from datetime import date, timedelta

SYSTEM = """You are DocFlow, an accounts-payable assistant for a small business.
Use the tools to look things up; never guess numbers, names or dates.
Today is {today}. For questions about amounts, vendors or due dates use list_invoices (call it with no status filter unless the user names a status).
To act on an invoice, pass its vendor name or invoice number; you do not need an internal id.
For questions about what a document says use search_documents and cite the filename.
You may approve invoices and draft emails when the user asks. Emails are saved as drafts, never sent.
Keep answers short and factual."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Semantic search over all uploaded documents (invoices, emails, contracts).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_invoices",
        "description": "List extracted invoices with vendor, number, dates, total and status.",
        "parameters": {"type": "object", "properties": {
            "vendor": {"type": "string", "description": "optional vendor name filter"},
            "status": {"type": "string", "enum": ["ready_to_pay", "awaiting_approval", "flagged", "approved"]},
            "due_within_days": {"type": "integer", "description": "only invoices due within this many days from today"}}}}},
    {"type": "function", "function": {
        "name": "approve_invoice",
        "description": "Mark an invoice as approved for payment.",
        "parameters": {"type": "object", "properties": {
            "invoice": {"type": "string", "description": "invoice number or vendor name"}}, "required": ["invoice"]}}},
    {"type": "function", "function": {
        "name": "draft_email",
        "description": "Save a draft email (for example a query to a vendor). It is not sent.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"},
            "invoice": {"type": "string", "description": "related invoice number or vendor, if any"}},
            "required": ["to", "subject", "body"]}}},
]


class Agent:
    def __init__(self, store, llm, max_steps=6):
        self.store, self.llm, self.max_steps = store, llm, max_steps

    def resolve(self, ref):
        """Find invoices by doc id, invoice number or vendor name (deterministic, not the LLM)."""
        ref = (ref or "").strip().lower()
        if not ref:
            return []
        invs = self.store.list_invoices()
        exact = [i for i in invs if ref in (i["doc_id"], (i["invoice_number"] or "").lower())]
        return exact or [i for i in invs if ref in (i["vendor"] or "").lower()
                         or (i["vendor"] or "").lower().split()[0] in ref]

    def run_tool(self, name, args, today=None):
        s = self.store
        today = today or date.today()
        if name == "search_documents":
            return s.search(args.get("query", ""))
        if name == "list_invoices":
            rows = s.list_invoices(args.get("vendor"), args.get("status"))
            days = args.get("due_within_days")
            if days is not None:
                limit = (today + timedelta(days=int(days))).isoformat()
                rows = [r for r in rows if r["due_date"] and today.isoformat() <= r["due_date"] <= limit]
            flags = {}
            for a in s.list_actions():
                if a["source"] == "workflow" and a["doc_id"]:
                    flags.setdefault(a["doc_id"], []).append(f"{a['type']}: {a['detail'].get('reason', '')}")
            return [{**{k: v for k, v in r.items() if k != "line_items"}, "flags": flags.get(r["doc_id"], [])}
                    for r in rows]
        if name == "approve_invoice":
            matches = self.resolve(args.get("invoice") or args.get("doc_id"))
            if len(matches) != 1:
                return {"error": "no matching invoice" if not matches else "more than one invoice matches",
                        "candidates": [{k: m[k] for k in ("vendor", "invoice_number", "status")} for m in matches]}
            doc_id = matches[0]["doc_id"]
            s.set_invoice_status(doc_id, "approved")
            s.add_action(doc_id, "approved", {"reason": "approved via assistant"}, source="agent")
            return {"ok": True, "invoice_number": matches[0]["invoice_number"], "status": "approved"}
        if name == "draft_email":
            detail = {k: args.get(k, "") for k in ("to", "subject", "body")}
            matches = self.resolve(args.get("invoice"))
            s.add_action(matches[0]["doc_id"] if matches else None, "email_draft", detail, source="agent")
            return {"ok": True, "saved_as": "draft"}
        return {"error": f"unknown tool {name}"}

    def ask(self, question, history=None):
        messages = [{"role": "system", "content": SYSTEM.format(today=date.today().isoformat())}, *(history or []),
                    {"role": "user", "content": question}]
        trace = []
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, tools=TOOLS)
            calls = reply.get("tool_calls") or []
            if not calls:
                return {"answer": reply.get("content", ""), "steps": trace}
            messages.append(reply)
            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result = self.run_tool(fn["name"], args)
                trace.append({"tool": fn["name"], "args": args, "result": result})
                messages.append({"role": "tool", "content": json.dumps(result, default=str)})
        return {"answer": "I stopped after too many steps. Try a more specific request.", "steps": trace}
