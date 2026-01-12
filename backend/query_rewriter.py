import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class QueryRewriter:
    """Rewrite user queries to improve retrieval quality"""

    def __init__(self):
        self.model = OLLAMA_MODEL

        self.prompt_template = """You are a search query optimizer.

Your job is to rewrite the user's question into a better search query that will retrieve relevant passages from documents.

Rules:
- Keep the meaning
- Expand abbreviations if needed
- Add relevant keywords
- Remove fluff
- Do NOT answer the question
- Output ONLY the rewritten query

User question:
{question}

Rewritten search query:"""

    def rewrite_query(self, question: str) -> str:
        try:
            question = question.strip()

            # Don't rewrite very short queries
            if len(question.split()) < 4:
                return question

            prompt = self.prompt_template.format(question=question)

            rewritten = self._call_ollama(prompt)

            rewritten = rewritten.strip()

            # Safety: fallback
            if len(rewritten) < 5:
                return question

            return rewritten

        except Exception as e:
            print("Query rewrite failed:", e)
            return question

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"]
