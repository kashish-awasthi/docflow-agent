"""Read uploaded files into text and split them into overlapping chunks."""
import email
import io
from email import policy

from pypdf import PdfReader


def read_file(filename, data: bytes):
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "pdf"
    if name.endswith(".eml"):
        msg = email.message_from_bytes(data, policy=policy.default)
        body = msg.get_body(preferencelist=("plain",))
        text = body.get_content() if body else ""
        header = f"From: {msg['from']}\nSubject: {msg['subject']}\nDate: {msg['date']}\n\n"
        return header + text, "email"
    return data.decode("utf-8", errors="replace"), "text"


def chunk(text, size=800, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return out
