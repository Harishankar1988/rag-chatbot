import os
import sys
import re
import json
import time
from typing import List, Dict, Any, Optional
#import google.generativeai as genai
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
import requests

class RAGSystem:
    """RAG system using Gemini LLM"""
    
    def __init__(self):
        # Configure Gemini
        #genai.configure(api_key=Config.GEMINI_API_KEY)
        #self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        

        
        # Enhanced system prompt for RAG with citation support
        self.system_prompt = """You are a helpful AI assistant that answers questions based ONLY on the provided context from PDF documents.

CRITICAL RULES:
1. Use ONLY the information from the provided context to answer questions
2. If the context doesn't contain the answer, say "I don't have enough information from the uploaded documents to answer this question"
3. Do not use any external knowledge or make up information
4. Provide specific citations for your answers using the format [Source X: Document Name, Page Y]
5. If multiple documents provide relevant information, synthesize them and cite all sources
6. Be thorough but concise - explain the reasoning when helpful
7. If you find conflicting information in different documents, acknowledge the conflict

CITATION EXAMPLES:
- "According to the research [Source 1: report.pdf, Page 3], the main cause is..."
- "Both documents agree on this point [Source 1: manual.pdf, Page 5] [Source 2: guide.pdf, Page 2]"
- "The procedure involves three steps [Source 1: instructions.pdf, Page 7]"

Context:
{context}

Question: {question}

Provide a detailed answer with proper citations:"""

    def _call_llm(self, prompt: str) -> str:
        """Call local Ollama LLM"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",   # or "qwen2.5" or "mistral"
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                },
                timeout=180
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise RuntimeError(f"Ollama LLM call failed: {e}")

    
    def _budget_context(self, documents: List[Document], max_tokens: int = 4000) -> List[Document]:
        """Budget context by token count instead of document count"""
        if not documents:
            return documents
        
        budgeted_docs = []
        current_tokens = 0
        
        # Rough token estimation (1 token ≈ 4 characters)
        for doc in documents:
            doc_tokens = len(doc.page_content) // 4
            
            if current_tokens + doc_tokens <= max_tokens:
                budgeted_docs.append(doc)
                current_tokens += doc_tokens
                doc.metadata['included_in_context'] = True
            else:
                # Truncate if it's the first document
                if not budgeted_docs:
                    remaining_tokens = max_tokens - current_tokens
                    truncated_content = doc.page_content[:remaining_tokens * 4]
                    doc.page_content = truncated_content
                    doc.metadata['truncated'] = True
                    budgeted_docs.append(doc)
                break
        
        return budgeted_docs
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string with enhanced metadata"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown Document')
            pages = doc.metadata.get('pages', [])
            primary_page = doc.metadata.get('primary_page')
            chunk_type = doc.metadata.get('chunk_type', 'paragraph')
            
            # Enhanced citation information
            page_info = f""
            if pages:
                if len(pages) == 1:
                    page_info = f" (Page {pages[0]})"
                else:
                    page_info = f" (Pages {min(pages)}-{max(pages)})"
            elif primary_page:
                page_info = f" (Page {primary_page})"
            
            chunk_info = f" [Chunk {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}, Type: {chunk_type}]"
            
            context_part = f"[Source {i}: {source}{page_info}{chunk_info}]\n{doc.page_content.strip()}"
            context_parts.append(context_part)
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_response(self, question: str, context_documents: List[Document]) -> Dict[str, Any]:
        """Generate response using RAG with confidence scoring"""
        try:
            # Sanitize question to prevent prompt injection
            question = self._sanitize_question(question)
            
            # Calculate confidence scores
            confidence_score = self._calculate_confidence(context_documents)
            
            # Check if we have enough relevant context
            if confidence_score < 0.3:
                return {
                    "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                    "sources": [],
                    "context_used": False,
                    "question": question,
                    "confidence_score": 0.0,
                    "not_found": True
                }
            
            # Budget context by token count
            budgeted_docs = self._budget_context(context_documents)
            
            # Format context
            context = self.format_context(budgeted_docs)
            
            # Create prompt with confidence awareness
            prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            # Generate response
            #response = self.model.generate_content(prompt)
            answer = self._call_llm(prompt)
            
            # Extract answer
            answer = response.text
            
            # Prepare enhanced response with detailed sources
            sources = []
            for i, doc in enumerate(context_documents, 1):
                source_info = {
                    "source_index": i,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "document_id": doc.metadata.get('document_id', ''),
                    "chunk_id": doc.metadata.get('chunk_id'),
                    "chunk_index": doc.metadata.get('chunk_index', 0),
                    "pages": doc.metadata.get('pages', []),
                    "primary_page": doc.metadata.get('primary_page'),
                    "chunk_type": doc.metadata.get('chunk_type', 'paragraph'),
                    "score": doc.metadata.get('hybrid_score', doc.metadata.get('score', 0)),
                    "vector_score": doc.metadata.get('vector_score', 0),
                    "keyword_score": doc.metadata.get('keyword_score', 0),
                    "search_type": doc.metadata.get('search_type', 'vector'),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "sentence_count": doc.metadata.get('sentence_count', 0),
                    "relevance_rank": i
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(context_documents) > 0,
                "question": question,
                "confidence_score": confidence_score,
                "not_found": False
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": False,
                "question": question,
                "error": str(e)
            }
    
    def _sanitize_question(self, question: str) -> str:
        """Sanitize question to prevent prompt injection"""
        # Remove potential prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard context",
            "forget everything",
            "system prompt",
            "act as",
            "pretend to be",
            "roleplay as",
            "new instructions:",
            "override:",
            "bypass:",
            "admin mode",
            "developer mode"
        ]
        
        question_lower = question.lower()
        for pattern in injection_patterns:
            if pattern in question_lower:
                question = question.replace(pattern, "")
        
        # Remove excessive special characters
        question = re.sub(r'[^\w\s\?\.\,\!\:\-\(\)]+', ' ', question)
        
        # Limit length
        question = question[:500].strip()
        
        return question
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on document relevance"""
        if not documents:
            return 0.0
        
        # Factor in multiple signals
        scores = []
        for doc in documents:
            # Use hybrid score if available, otherwise fallback to regular score
            score = doc.metadata.get('hybrid_score', doc.metadata.get('score', 0))
            scores.append(score)
        
        if not scores:
            return 0.0
        
        # Calculate weighted confidence
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # Boost confidence if we have high-scoring documents
        confidence = (avg_score * 0.7) + (max_score * 0.3)
        
        # Normalize to 0-1 range (rough approximation)
        return min(1.0, confidence / 2.0)
    
    def chat_with_history(self, question: str, context_documents: List[Document], 
                         chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Chat with conversation history and confidence scoring"""
        try:
            # Calculate confidence scores
            confidence_score = self._calculate_confidence(context_documents)
            
            # Check if we have enough relevant context
            if confidence_score < 0.3:
                return {
                    "answer": "I couldn't find relevant information in the uploaded documents to answer this question.",
                    "sources": [],
                    "context_used": False,
                    "question": question,
                    "confidence_score": 0.0,
                    "not_found": True
                }
            
            # Format context
            context = self.format_context(context_documents)
            
            # Build conversation history
            history_text = ""
            if chat_history:
                history_parts = []
                for msg in chat_history[-5:]:  # Keep last 5 messages for context
                    if msg['role'] == 'user':
                        history_parts.append(f"User: {msg['content']}")
                    else:
                        history_parts.append(f"Assistant: {msg['content']}")
                history_text = "\n".join(history_parts) + "\n\n"
            
            # Enhanced prompt with history and citation requirements
            enhanced_prompt = f"""You are a helpful AI assistant that answers questions based ONLY on the provided context from PDF documents.

Previous conversation:
{history_text}

Current context from documents:
{context}

Current question: {question}

CRITICAL RULES:
1. Use ONLY the information from the provided context to answer questions
2. Consider the conversation history for context but still base answers on document content
3. If the context doesn't contain the answer, say "I don't have enough information from the uploaded documents to answer this question"
4. Do not use any external knowledge or make up information
5. Provide specific citations for your answers using the format [Source X: Document Name, Page Y]
6. If multiple documents provide relevant information, synthesize them and cite all sources
7. Be thorough but concise - explain the reasoning when helpful
8. If you find conflicting information in different documents, acknowledge the conflict

CITATION EXAMPLES:
- "According to the research [Source 1: report.pdf, Page 3], the main cause is..."
- "Both documents agree on this point [Source 1: manual.pdf, Page 5] [Source 2: guide.pdf, Page 2]"
- "The procedure involves three steps [Source 1: instructions.pdf, Page 7]"

Provide a detailed answer with proper citations:"""
            
            # Generate response
            #response = self.model.generate_content(enhanced_prompt)
            #answer = response.text
            answer = self._call_llm(enhanced_prompt)
            
            # Prepare enhanced response with detailed sources
            sources = []
            for i, doc in enumerate(context_documents, 1):
                source_info = {
                    "source_index": i,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "document_id": doc.metadata.get('document_id', ''),
                    "chunk_id": doc.metadata.get('chunk_id'),
                    "chunk_index": doc.metadata.get('chunk_index', 0),
                    "pages": doc.metadata.get('pages', []),
                    "primary_page": doc.metadata.get('primary_page'),
                    "chunk_type": doc.metadata.get('chunk_type', 'paragraph'),
                    "score": doc.metadata.get('hybrid_score', doc.metadata.get('score', 0)),
                    "vector_score": doc.metadata.get('vector_score', 0),
                    "keyword_score": doc.metadata.get('keyword_score', 0),
                    "search_type": doc.metadata.get('search_type', 'vector'),
                    "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "sentence_count": doc.metadata.get('sentence_count', 0),
                    "relevance_rank": i
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(context_documents) > 0,
                "question": question,
                "confidence_score": confidence_score,
                "not_found": False
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": False,
                "question": question,
                "error": str(e)
            }
