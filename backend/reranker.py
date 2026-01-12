from typing import List
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document

class ChunkReranker:
    """Cross-encoder based reranker"""

    def __init__(self):
        print("Loading cross-encoder reranker...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)

        # Attach scores
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        # Sort by rerank score
        documents = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)

        # Assign rerank positions
        for i, doc in enumerate(documents):
            doc.metadata["rerank_position"] = i + 1

        return documents[:top_k]
