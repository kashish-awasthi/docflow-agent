"""Storage: ChromaDB for document chunks (vector search), SQLite for structured records."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, filename TEXT, kind TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS invoices (
  doc_id TEXT PRIMARY KEY, vendor TEXT, invoice_number TEXT, invoice_date TEXT,
  due_date TEXT, currency TEXT, total REAL, line_items TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, type TEXT, detail TEXT,
  created_at TEXT, source TEXT);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir="data", embed_fn=None):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(Path(data_dir) / "docflow.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.chroma = chromadb.PersistentClient(path=str(Path(data_dir) / "chroma"))
        self.chunks = self.chroma.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
        self.embed_fn = embed_fn

    # documents and chunks
    def add_document(self, filename, kind, text, chunks):
        doc_id = uuid.uuid4().hex[:8]
        self.db.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (doc_id, filename, kind, text, now()))
        self.db.commit()
        if chunks:
            self.chunks.add(
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=self.embed_fn(chunks),
                metadatas=[{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))],
            )
        return doc_id

    def get_document(self, doc_id):
        row = self.db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self):
        return [dict(r) for r in self.db.execute(
            "SELECT id, filename, kind, created_at FROM documents ORDER BY created_at")]

    def search(self, query, k=4):
        if self.chunks.count() == 0:
            return []
        res = self.chunks.query(query_embeddings=self.embed_fn([query]), n_results=min(k, self.chunks.count()))
        return [{"doc_id": m["doc_id"], "filename": m["filename"], "text": d, "score": round(1 - dist, 3)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    # invoices
    def save_invoice(self, doc_id, inv, status="extracted"):
        self.db.execute(
            "INSERT OR REPLACE INTO invoices VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, inv.get("vendor"), inv.get("invoice_number"), inv.get("invoice_date"),
             inv.get("due_date"), inv.get("currency"), inv.get("total"),
             json.dumps(inv.get("line_items", [])), status))
        self.db.commit()

    def set_invoice_status(self, doc_id, status):
        self.db.execute("UPDATE invoices SET status=? WHERE doc_id=?", (status, doc_id))
        self.db.commit()

    def list_invoices(self, vendor=None, status=None):
        sql, args = "SELECT * FROM invoices WHERE 1=1", []
        if vendor:
            sql += " AND lower(vendor) LIKE ?"
            args.append(f"%{vendor.lower()}%")
        if status:
            sql += " AND status=?"
            args.append(status)
        rows = [dict(r) for r in self.db.execute(sql + " ORDER BY due_date", args)]
        for r in rows:
            r["line_items"] = json.loads(r["line_items"] or "[]")
        return rows

    def find_invoice_number(self, vendor, number, exclude_doc):
        return self.db.execute(
            "SELECT doc_id FROM invoices WHERE lower(vendor)=lower(?) AND invoice_number=? AND doc_id<>?",
            (vendor or "", number or "", exclude_doc)).fetchone()

    # actions
    def add_action(self, doc_id, type_, detail, source="workflow"):
        self.db.execute("INSERT INTO actions (doc_id, type, detail, created_at, source) VALUES (?,?,?,?,?)",
                        (doc_id, type_, json.dumps(detail), now(), source))
        self.db.commit()

    def list_actions(self):
        rows = [dict(r) for r in self.db.execute("SELECT * FROM actions ORDER BY id DESC")]
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows
