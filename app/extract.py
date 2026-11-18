"""LLM-based invoice extraction, followed by deterministic validation."""
from datetime import date

from .llm import parse_json

PROMPT = """You extract data from business documents. Reply with JSON only.
Decide whether the document is an invoice. If it is, return:
{"is_invoice": true, "vendor": str, "invoice_number": str, "invoice_date": "YYYY-MM-DD",
 "due_date": "YYYY-MM-DD" or null, "currency": "INR"|"USD"|..., "total": number,
 "line_items": [{"description": str, "quantity": number, "amount": number}]}
If it is not an invoice, return {"is_invoice": false, "summary": str}.
Copy values exactly from the document. Never invent values; use null when missing.

Document:
"""


def _valid_date(s):
    try:
        return date.fromisoformat(s).isoformat() if s else None
    except (TypeError, ValueError):
        return None


def _num(x):
    try:
        return round(float(str(x).replace(",", "")), 2)
    except (TypeError, ValueError):
        return None


def validate(raw):
    """Normalise model output and list problems a human should check."""
    inv = {
        "vendor": (raw.get("vendor") or "").strip() or None,
        "invoice_number": str(raw.get("invoice_number") or "").strip() or None,
        "invoice_date": _valid_date(raw.get("invoice_date")),
        "due_date": _valid_date(raw.get("due_date")),
        "currency": (raw.get("currency") or "INR").upper(),
        "total": _num(raw.get("total")),
        "line_items": [],
    }
    for item in raw.get("line_items") or []:
        if isinstance(item, dict):
            inv["line_items"].append({"description": str(item.get("description", "")),
                                      "quantity": _num(item.get("quantity")),
                                      "amount": _num(item.get("amount"))})
    issues = [f"missing {f}" for f in ("vendor", "invoice_number", "total") if not inv[f]]
    amounts = [i["amount"] for i in inv["line_items"] if i["amount"] is not None]
    if inv["total"] and amounts and abs(sum(amounts) - inv["total"]) > max(1, 0.02 * inv["total"]):
        issues.append(f"line items add up to {sum(amounts):.2f}, not the total {inv['total']:.2f}")
    return inv, issues


def extract(llm, text):
    reply = llm.chat([{"role": "user", "content": PROMPT + text[:6000]}], json_mode=True)
    raw = parse_json(reply["content"])
    if not raw.get("is_invoice"):
        return None, [], raw.get("summary", "")
    inv, issues = validate(raw)
    return inv, issues, ""
