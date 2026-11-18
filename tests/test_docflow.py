"""Tests run without Ollama: a fake LLM returns fixed replies, and a fake embedder makes vectors from words."""
import hashlib
import json
from datetime import date

from app.agent import Agent
from app.extract import validate
from app.ingest import chunk, read_file
from app.store import Store
from app.workflow import process_upload

TODAY = date(2026, 11, 18)


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v if any(v) else [1.0] + [0.0] * 63)
    return out


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append(messages)
        r = self.replies.pop(0)
        return {"content": json.dumps(r)} if json_mode else r


def inv(**kw):
    base = {"is_invoice": True, "vendor": "Acme", "invoice_number": "A-1", "invoice_date": "2026-11-10",
            "due_date": "2026-12-30", "currency": "INR", "total": 1000,
            "line_items": [{"description": "x", "quantity": 1, "amount": 1000}]}
    return {**base, **kw}


def store(tmp_path):
    return Store(str(tmp_path), embed_fn=fake_embed)


def test_chunk_overlaps_and_covers_text():
    parts = chunk("word " * 500, size=200, overlap=50)
    assert len(parts) > 1 and all(len(p) <= 200 for p in parts)


def test_read_email_includes_headers():
    raw = b"From: a@b.c\nSubject: Hi\nContent-Type: text/plain\n\nBody text"
    text, kind = read_file("m.eml", raw)
    assert kind == "email" and "Subject: Hi" in text and "Body text" in text


def test_validate_normalises_and_flags_mismatch():
    v, issues = validate(inv(total="1,500.00", invoice_date="10/11/2026"))
    assert v["total"] == 1500.0 and v["invoice_date"] is None
    assert any("add up" in i for i in issues)


def test_validate_flags_missing_fields():
    _, issues = validate(inv(vendor="", total=None))
    assert "missing vendor" in issues and "missing total" in issues


def test_clean_invoice_is_ready_to_pay(tmp_path):
    s = store(tmp_path)
    r = process_upload(s, FakeLLM([inv()]), "a.txt", b"invoice text", today=TODAY)
    assert r["status"] == "ready_to_pay" and r["actions"] == []


def test_large_invoice_needs_approval(tmp_path):
    s = store(tmp_path)
    r = process_upload(s, FakeLLM([inv(total=90000, line_items=[])]), "a.txt", b"x", today=TODAY)
    assert r["status"] == "awaiting_approval"


def test_due_soon_and_overdue(tmp_path):
    s = store(tmp_path)
    r1 = process_upload(s, FakeLLM([inv(due_date="2026-11-22")]), "a.txt", b"x", today=TODAY)
    r2 = process_upload(s, FakeLLM([inv(invoice_number="A-2", due_date="2026-11-01")]), "b.txt", b"y", today=TODAY)
    assert r1["actions"][0]["type"] == "payment_reminder"
    assert r2["actions"][0]["type"] == "overdue"


def test_duplicate_invoice_is_flagged(tmp_path):
    s = store(tmp_path)
    process_upload(s, FakeLLM([inv()]), "a.txt", b"x", today=TODAY)
    r = process_upload(s, FakeLLM([inv()]), "a2.txt", b"x again", today=TODAY)
    assert r["status"] == "flagged" and any(a["type"] == "duplicate" for a in r["actions"])


def test_non_invoice_is_filed_and_searchable(tmp_path):
    s = store(tmp_path)
    r = process_upload(s, FakeLLM([{"is_invoice": False, "summary": "bank change"}]), "m.txt",
                       b"Our new bank account is at ICICI", today=TODAY)
    assert r["status"] == "filed"
    assert s.search("new bank account ICICI")[0]["filename"] == "m.txt"


def test_agent_calls_tool_then_answers(tmp_path):
    s = store(tmp_path)
    process_upload(s, FakeLLM([inv(total=90000, line_items=[])]), "a.txt", b"x", today=TODAY)
    llm = FakeLLM([
        {"content": "", "tool_calls": [{"function": {"name": "approve_invoice", "arguments": {"invoice": "acme"}}}]},
        {"content": "Approved."},
    ])
    out = Agent(s, llm).ask("approve it")
    assert out["answer"] == "Approved." and out["steps"][0]["result"]["status"] == "approved"
    assert s.list_invoices()[0]["status"] == "approved"


def test_agent_rejects_unknown_invoice(tmp_path):
    s = store(tmp_path)
    assert "error" in Agent(s, None).run_tool("approve_invoice", {"invoice": "nope"})


def test_due_within_days_filter(tmp_path):
    s = store(tmp_path)
    process_upload(s, FakeLLM([inv(due_date="2026-11-22")]), "a.txt", b"x", today=TODAY)
    process_upload(s, FakeLLM([inv(invoice_number="A-2", due_date="2026-12-30")]), "b.txt", b"y", today=TODAY)
    rows = Agent(s, None).run_tool("list_invoices", {"due_within_days": 7}, today=TODAY)
    assert [r["invoice_number"] for r in rows] == ["A-1"]


def test_draft_email_is_saved_not_sent(tmp_path):
    s = store(tmp_path)
    Agent(s, None).run_tool("draft_email", {"to": "v@x.com", "subject": "Hi", "body": "b"})
    assert s.list_actions()[0]["type"] == "email_draft"
