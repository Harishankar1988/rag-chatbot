import os
import sys
import re
import time
import json
import requests
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class RAGSystem:
    """RAG system using local Ollama (LLaMA)"""

    def __init__(self):
        self.model = OLLAMA_MODEL

        self.system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context from PDF documents.

CRITICAL RULES:
1. Use ONLY the information from the provided context to answer questions
2. If the context doesn't contain the answer, say: "I don't have enough information from the uploaded documents to answer this question."
3. Do NOT use any external knowledge
4. Always cite sources using: [Source X: filename, Page Y]
5. If multiple sources are used, cite all
6. Be clear and concise

Context:
{context}

Question: {question}

Answer with citations:"""

    # ---------------------------
    # Public entry point
    # ---------------------------
    def chat_with_history(self, question: str, context_documents: List[Document], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        question = question.strip()

        # 1. Small talk / meta intent
        smalltalk = self._handle_smalltalk(question)
        if smalltalk:
            return {
                "answer": smalltalk,
                "sources": [],
                "context_used": False,
                "confidence_score": 1.0,
                "not_found": False
            }

        # 2. Sanitize
        question = self._sanitize_question(question)

        # 3. Confidence gate
        confidence = self._calculate_confidence(context_documents)
        if confidence < 0.3:
            return {
                "answer": "I don't have enough information from the uploaded documents to answer this question.",
                "sources": [],
                "context_used": False,
                "confidence_score": 0.0,
                "not_found": True
            }

        # 4. Budget context
        budgeted_docs = self._budget_context(context_documents)

        # 5. Format context
        context = self.format_context(budgeted_docs)

        # 6. Build prompt
        prompt = self.system_prompt.format(context=context, question=question)

        # 7. Call Ollama
        answer = self._call_ollama(prompt)

        # 8. Build sources
        sources = self._build_sources(context_documents)

        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "confidence_score": confidence,
            "not_found": False
        }

    # ---------------------------
    # Ollama call
    # ---------------------------
    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["response"]

    # ---------------------------
    # Small talk router
    # ---------------------------
    def _handle_smalltalk(self, question: str) -> Optional[str]:
        q = question.lower()

        greetings = ["hi", "hello", "hey"]
        if q in greetings:
            return "Hi! I can answer questions about the PDFs you have uploaded."

        if "who are you" in q:
            return "I'm a document assistant. I answer questions using your uploaded PDFs."

        if "what can you do" in q:
            return "You can upload PDFs and ask me questions about their content."

        return None

    # ---------------------------
    # Context formatting
    # ---------------------------
    def format_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            pages = doc.metadata.get("pages", [])
            page_str = f"Page {pages[0]}" if pages else "Unknown page"
            parts.append(f"[Source {i}: {source}, {page_str}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ---------------------------
    # Context budgeting
    # ---------------------------
    def _budget_context(self, documents: List[Document], max_tokens: int = 3500) -> List[Document]:
        budget = []
        tokens = 0

        for doc in documents:
            size = len(doc.page_content) // 4
            if tokens + size > max_tokens:
                break
            budget.append(doc)
            tokens += size

        return budget

    # ---------------------------
    # Confidence scoring
    # ---------------------------
    def _calculate_confidence(self, documents: List[Document]) -> float:
        if not documents:
            return 0.0

        scores = []
        for doc in documents:
            s = doc.metadata.get("hybrid_score", doc.metadata.get("score", 0))
            scores.append(s)

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores)
        mx = max(scores)

        confidence = (avg * 0.7) + (mx * 0.3)
        return min(1.0, confidence / 2.0)

    # ---------------------------
    # Sanitization
    # ---------------------------
    def _sanitize_question(self, question: str) -> str:
        bad = [
            "ignore previous instructions",
            "system prompt",
            "act as",
            "developer mode",
            "jailbreak"
        ]
        q = question.lower()
        for b in bad:
            q = q.replace(b, "")
        q = re.sub(r"[^\w\s\?\.\,\!\-\(\)]+", " ", q)
        return q[:500].strip()

    # ---------------------------
    # Source builder
    # ---------------------------
    def _build_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append({
                "source_index": i,
                "source": doc.metadata.get("source"),
                "pages": doc.metadata.get("pages"),
                "score": doc.metadata.get("hybrid_score", doc.metadata.get("score", 0)),
                "preview": doc.page_content[:200]
            })
        return sources
