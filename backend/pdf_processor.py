import os
import uuid
import re
from typing import List, Dict, Any, Tuple
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import nltk
from nltk.tokenize import sent_tokenize
import hashlib

class PDFProcessor:
    """Handles PDF processing, text extraction, and semantic chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Enhanced text splitter with semantic awareness
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from PDF file with page-level tracking"""
        pages_text = []
        full_text = ""
        
        # Try with pdfplumber first (better for tables and layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append({
                            "page_number": page_num,
                            "text": page_text,
                            "char_count": len(page_text)
                        })
                        full_text += f"\n--- Page {page_num} ---\n{page_text}\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append({
                                "page_number": page_num,
                                "text": page_text,
                                "char_count": len(page_text)
                            })
                            full_text += f"\n--- Page {page_num} ---\n{page_text}\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise Exception(f"Could not extract text from PDF: {pdf_path}")
        
        return full_text.strip(), pages_text
    
    def process_pdf(self, pdf_path: str, filename: str = None) -> List[Document]:
        """Process PDF file and return chunked documents with enhanced metadata"""
        if not filename:
            filename = os.path.basename(pdf_path)
        
        # Generate document-level ID
        doc_id = hashlib.md5(f"{filename}_{os.path.getsize(pdf_path)}".encode()).hexdigest()[:12]
        
        # Extract text with page information
        text, pages_info = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {filename}")
        
        # Create base metadata
        base_metadata = {
            "document_id": doc_id,
            "source": filename,
            "file_path": pdf_path,
            "file_type": "pdf",
            "total_chars": len(text),
            "total_pages": len(pages_info),
            "pages_info": pages_info
        }
        
        # Use semantic chunking
        documents = self._semantic_chunking(text, base_metadata, filename)
        
        return documents
    
    def _semantic_chunking(self, text: str, base_metadata: Dict[str, Any], filename: str) -> List[Document]:
        """Advanced semantic chunking with page mapping"""
        documents = []
        
        # Split text into chunks using enhanced splitter
        chunks = self.text_splitter.split_text(text)
        
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            # Skip empty or very small chunks
            chunk_clean = chunk.strip()
            if len(chunk_clean) < 50:  # Skip chunks smaller than 50 characters
                continue
            
            if not chunk_clean or chunk_clean.isspace():  # Skip empty/whitespace chunks
                continue
            
            valid_chunks.append(chunk)
            
            # Extract pages that this chunk belongs to
            chunk_pages = self._extract_pages_from_chunk(chunk, base_metadata.get('pages_info', []))
            
            # Create enhanced metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": len(valid_chunks) - 1,  # Use actual index after filtering
                "total_chunks": len(chunks),  # Keep original count for reference
                "chunk_chars": len(chunk),
                "pages": chunk_pages,
                "primary_page": chunk_pages[0] if chunk_pages else None,
                "sentence_count": len(sent_tokenize(chunk)),
                "paragraph_count": len([p for p in chunk.split('\n\n') if p.strip()]),
                "chunk_type": self._classify_chunk_type(chunk)
            })
            
            doc = Document(
                page_content=chunk_clean,  # Use cleaned chunk
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def _extract_pages_from_chunk(self, chunk: str, pages_info: List[Dict[str, Any]]) -> List[int]:
        """Extract page numbers from chunk based on page markers"""
        pages = []
        
        # Look for page markers in the chunk
        page_pattern = r'--- Page (\d+) ---'
        matches = re.findall(page_pattern, chunk)
        
        if matches:
            pages = [int(match) for match in matches]
        else:
            # Fallback: estimate page based on text position
            if pages_info:
                # Simple heuristic: assume even distribution
                estimated_page = max(1, min(len(pages_info), 
                                           len(chunk) // (len(pages_info[0].get('text', '')) or 1)))
                pages = [estimated_page]
        
        return sorted(list(set(pages)))
    
    def _classify_chunk_type(self, chunk: str) -> str:
        """Classify the type of content in the chunk"""
        chunk_lower = chunk.lower()
        
        # Check for headings
        if re.match(r'^#{1,6}\s+', chunk) or (len(chunk.split('\n')) == 1 and len(chunk) < 100):
            return "heading"
        
        # Check for lists
        if re.match(r'^\s*[-*+•]\s+', chunk, re.MULTILINE) or re.match(r'^\s*\d+\.\s+', chunk, re.MULTILINE):
            return "list"
        
        # Check for tables (simple heuristic)
        if '|' in chunk and chunk.count('|') > 3:
            return "table"
        
        # Check for questions
        if '?' in chunk and any(q in chunk_lower for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            return "question"
        
        # Default to paragraph
        return "paragraph"
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        """Process multiple PDF files"""
        all_documents = []
        
        for pdf_path in pdf_paths:
            try:
                filename = os.path.basename(pdf_path)
                documents = self.process_pdf(pdf_path, filename)
                all_documents.extend(documents)
                print(f"Successfully processed {filename}: {len(documents)} chunks")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        return all_documents
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get basic information about PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = {
                    "filename": os.path.basename(pdf_path),
                    "file_size": os.path.getsize(pdf_path),
                    "num_pages": len(pdf_reader.pages),
                    "title": pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                    "author": pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown'
                }
                return info
        except Exception as e:
            return {"error": str(e)}
