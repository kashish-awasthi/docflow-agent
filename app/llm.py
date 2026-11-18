"""Thin client for a local Ollama server: chat (with tool calling) and embeddings."""
import json
import os

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


class OllamaLLM:
    def __init__(self, url=OLLAMA_URL, chat_model=CHAT_MODEL, embed_model=EMBED_MODEL):
        self.url = url
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.http = httpx.Client(timeout=180)

    def chat(self, messages, tools=None, json_mode=False):
        """Returns the assistant message dict: {"content": str, "tool_calls": [...]}."""
        body = {"model": self.chat_model, "messages": messages, "stream": False,
                "options": {"temperature": 0}}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["format"] = "json"
        r = self.http.post(f"{self.url}/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]

    def embed(self, texts):
        r = self.http.post(f"{self.url}/api/embed", json={"model": self.embed_model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def available(self):
        try:
            return self.http.get(f"{self.url}/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False


def parse_json(text):
    """Parse JSON from a model reply, tolerating text around the object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])
