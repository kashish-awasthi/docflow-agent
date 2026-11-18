"""The automatic pipeline that runs on every upload: ingest -> extract -> business rules -> actions."""
from datetime import date

from .extract import extract
from .ingest import chunk, read_file

APPROVAL_LIMIT = 50000  # invoices above this need a manager's approval
DUE_SOON_DAYS = 7


def apply_rules(store, doc_id, inv, issues, today=None):
    today = today or date.today()
    actions = []

    def act(type_, detail):
        store.add_action(doc_id, type_, detail)
        actions.append({"type": type_, **detail})

    if issues:
        act("needs_review", {"reason": "; ".join(issues)})
    if inv["vendor"] and inv["invoice_number"] and store.find_invoice_number(
            inv["vendor"], inv["invoice_number"], doc_id):
        act("duplicate", {"reason": f"invoice {inv['invoice_number']} from {inv['vendor']} was already received"})
    if inv["total"] and inv["total"] > APPROVAL_LIMIT:
        act("approval_required", {"reason": f"total {inv['total']:,.2f} is above {APPROVAL_LIMIT:,}"})
    if inv["due_date"]:
        days = (date.fromisoformat(inv["due_date"]) - today).days
        if days < 0:
            act("overdue", {"reason": f"was due {-days} days ago"})
        elif days <= DUE_SOON_DAYS:
            act("payment_reminder", {"reason": f"due in {days} days"})

    status = "flagged" if any(a["type"] in ("needs_review", "duplicate") for a in actions) else \
        "awaiting_approval" if any(a["type"] == "approval_required" for a in actions) else "ready_to_pay"
    store.set_invoice_status(doc_id, status)
    return status, actions


def process_upload(store, llm, filename, data, today=None):
    text, kind = read_file(filename, data)
    doc_id = store.add_document(filename, kind, text, chunk(text))
    result = {"doc_id": doc_id, "filename": filename, "kind": kind}
    inv, issues, summary = extract(llm, text)
    if inv is None:
        store.add_action(doc_id, "filed", {"reason": "not an invoice", "summary": summary})
        result.update(invoice=None, summary=summary, status="filed", actions=[])
        return result
    store.save_invoice(doc_id, inv)
    status, actions = apply_rules(store, doc_id, inv, issues, today)
    result.update(invoice=inv, issues=issues, status=status, actions=actions)
    return result
