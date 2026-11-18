# DocFlow Agent

A small accounts-payable tool I built to try out agents, vector search and workflow automation on a real-ish problem.

You upload invoices and vendor emails. It reads them, pulls out the invoice details, runs some business checks, and you can ask it things like "which invoices are due this week?" or tell it to "approve the Bright Office invoice".

Everything runs locally and free: Llama 3.1 8B through Ollama, nomic-embed-text for embeddings, and ChromaDB as the vector store.

## How it works

```
upload -> read (pdf / eml / txt) -> chunks -> embeddings -> ChromaDB
                                 -> LLM extracts invoice JSON -> validation -> SQLite -> rules -> actions

question -> agent loop (LLM picks a tool, gets result, repeats) -> answer
```

**Extraction.** The LLM returns JSON (vendor, invoice number, dates, total, line items). I don't trust it blindly, so the output goes through validation: dates and amounts are normalised, missing fields are reported, and if the line items don't add up to the total the invoice goes to review.

**Rules** run on every invoice:
- same vendor + invoice number seen before -> duplicate
- total over 50,000 -> needs approval
- due in the next 7 days -> payment reminder, past due -> overdue

**Agent tools:**
- `search_documents` - semantic search over everything uploaded
- `list_invoices` - filter by vendor, status or "due within N days"
- `approve_invoice`
- `draft_email` - saved as a draft, never actually sent

Invoices are looked up by code from the vendor name or number the model gives, so the model can't approve the wrong one by making up an ID. The UI shows each tool call so you can see what the agent did.

## Sample run

The `samples/` folder has 4 invoices and 1 email. After uploading them:

| File | Result |
|---|---|
| acme_invoice_1042 | ready to pay, reminder (due in 4 days) |
| acme_invoice_1042_resent | flagged as duplicate |
| bright_office_inv_778 (Rs 1,34,000) | waiting for approval |
| quickprint_invoice_q55 | flagged - items add up to 15,000 but total says 18,500, also overdue |
| vendor_email.eml | stored, searchable |

Some questions I tried:

- "Which invoices are due in the next week?" -> calls `list_invoices(due_within_days=7)` -> ACM-1042, due 22 Nov
- "Approve the Bright Office invoice" -> `approve_invoice("Bright Office")` -> BOF-778 approved
- "Did Acme send anything about bank details?" -> `search_documents` -> finds the email about the new bank account and 2% early payment discount

## Running it

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
ollama serve

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.api:app --port 8765
```

Then open http://localhost:8765 and drag in the files from `samples/`.

Env vars if you want to change things: `CHAT_MODEL`, `EMBED_MODEL`, `OLLAMA_URL`, `DATA_DIR`.

## Tests

```bash
DOCFLOW_NO_AUTOAPP=1 .venv/bin/python -m pytest -q
```

13 tests. They use a fake LLM and fake embeddings so they don't need Ollama running.

## Code

- `app/ingest.py` - reading files, chunking
- `app/store.py` - ChromaDB + SQLite
- `app/extract.py` - LLM extraction and validation
- `app/workflow.py` - upload pipeline and rules
- `app/agent.py` - agent loop and tools
- `app/api.py` - FastAPI routes
- `app/static/index.html` - frontend

## Known issues / todo

- The 8B model sometimes uses a narrower filter than it should (e.g. only looks at flagged invoices when asked what's overdue). A bigger model handles this better.
- No OCR, so scanned PDFs won't work.
- Approve and email are only recorded, nothing is paid or sent.
