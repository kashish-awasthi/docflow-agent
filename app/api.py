"""FastAPI server: upload documents, see invoices and actions, chat with the agent."""
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agent import Agent
from .llm import OllamaLLM
from .store import Store
from .workflow import process_upload


def create_app(store=None, llm=None):
    llm = llm or OllamaLLM()
    store = store or Store(os.getenv("DATA_DIR", "data"), embed_fn=llm.embed)
    agent = Agent(store, llm)
    app = FastAPI(title="DocFlow Agent")
    static = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/status")
    def status():
        return {"llm": llm.available(), "documents": len(store.list_documents())}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        try:
            return process_upload(store, llm, file.filename, data)
        except Exception as e:  # surface model or parsing failures to the UI
            raise HTTPException(500, f"processing failed: {e}")

    @app.get("/api/documents")
    def documents():
        return store.list_documents()

    @app.get("/api/invoices")
    def invoices(vendor: str | None = None, status: str | None = None):
        return store.list_invoices(vendor, status)

    @app.get("/api/actions")
    def actions():
        return store.list_actions()

    class Ask(BaseModel):
        question: str

    @app.post("/api/ask")
    def ask(body: Ask):
        if not body.question.strip():
            raise HTTPException(400, "empty question")
        return agent.ask(body.question)

    return app


app = create_app() if os.getenv("DOCFLOW_NO_AUTOAPP") is None else None
