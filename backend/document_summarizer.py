import os
import sys
import json
from typing import List, Dict, Any
import google.generativeai as genai
from langchain.docstore.document import Document

# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class DocumentSummarizer:
    """Generate document-level summaries and key topics"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
        # System prompts
        self.summary_prompt = """Generate a concise summary and extract key topics from the following document content.

Document content:
{content}

Please provide:
1. A 2-3 sentence summary
2. 3-5 key topics or themes
3. Document type (report, manual, research, etc.)

Format your response as JSON:
{{
    "summary": "Brief summary of the document",
    "key_topics": ["topic1", "topic2", "topic3"],
    "document_type": "manual",
    "word_count": 1500,
    "complexity": "medium"
}}"""
        
        self.batch_summary_prompt = """Generate a collection overview from multiple document summaries.

Document summaries:
{summaries}

Provide:
1. Overall collection summary
2. Common themes across documents
3. Document categories

Format as JSON:
{{
    "collection_summary": "Overview of all documents",
    "common_themes": ["theme1", "theme2"],
    "categories": {{"category": "count"}, {"category": "count"}}
}}"""
    
    def summarize_document(self, content: str, filename: str = "") -> Dict[str, Any]:
        """Generate summary for a single document"""
        try:
            # Truncate content if too long
            max_content = 8000  # Leave room for prompt
            if len(content) > max_content:
                content = content[:max_content] + "..."
            
            prompt = self.summary_prompt.format(content=content)
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            import json
            summary_data = json.loads(response.text.strip())
            
            # Add metadata
            summary_data.update({
                "filename": filename,
                "generated_at": str(genai.__version__),
                "content_length": len(content)
            })
            
            return summary_data
            
        except Exception as e:
            print(f"Error summarizing document {filename}: {e}")
            return {
                "summary": f"Error generating summary for {filename}",
                "key_topics": [],
                "document_type": "unknown",
                "filename": filename,
                "error": str(e)
            }
    
    def summarize_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """Generate summaries for multiple documents"""
        summaries = []
        
        for doc in documents:
            # Get document content (combine chunks from same document)
            doc_id = doc.metadata.get('document_id')
            if not doc_id:
                continue
                
            # Check if we already processed this document
            if any(s.get('document_id') == doc_id for s in summaries):
                continue
            
            summary = self.summarize_document(
                doc.page_content[:5000],  # Limit content for summary
                doc.metadata.get('source', 'Unknown')
            )
            summary['document_id'] = doc_id
            summaries.append(summary)
        
        # Generate collection overview
        try:
            summaries_text = "\n\n".join([
                f"- {s.get('filename', 'Unknown')}: {s.get('summary', 'No summary')}"
                for s in summaries[:10]  # Limit to 10 for context
            ])
            
            prompt = self.batch_summary_prompt.format(summaries=summaries_text)
            response = self.model.generate_content(prompt)
            
            import json
            collection_summary = json.loads(response.text.strip())
            
            return {
                "document_summaries": summaries,
                "collection_summary": collection_summary,
                "total_documents": len(summaries),
                "generated_at": str(genai.__version__)
            }
            
        except Exception as e:
            print(f"Error generating batch summary: {e}")
            return {
                "document_summaries": summaries,
                "collection_summary": {
                    "collection_summary": "Error generating collection overview",
                    "common_themes": [],
                    "categories": {}
                },
                "error": str(e)
            }
    
    def extract_key_terms(self, content: str, max_terms: int = 20) -> List[str]:
        """Extract key terms and phrases from document"""
        try:
            prompt = f"""Extract the most important terms, acronyms, and technical phrases from this content.

Content:
{content[:4000]}

Provide up to {max_terms} key terms that would be useful for search and retrieval.
Format as a JSON list: ["term1", "term2", "term3"]

Key terms:"""
            
            response = self.model.generate_content(prompt)
            
            import json
            terms = json.loads(response.text.strip())
            
            return terms if isinstance(terms, list) else []
            
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []
    
    def categorize_document(self, content: str, filename: str = "") -> Dict[str, Any]:
        """Categorize document by type and purpose"""
        try:
            prompt = f"""Analyze this document and categorize it.

Filename: {filename}
Content: {content[:3000]}

Determine:
1. Document type (manual, report, research, specification, guide, tutorial, etc.)
2. Primary purpose (informative, instructional, reference, analytical)
3. Technical level (beginner, intermediate, advanced)
4. Target audience (developers, users, managers, researchers)

Format as JSON:
{{
    "document_type": "manual",
    "primary_purpose": "instructional",
    "technical_level": "intermediate",
    "target_audience": "users"
}}"""
            
            response = self.model.generate_content(prompt)
            
            import json
            categorization = json.loads(response.text.strip())
            categorization['filename'] = filename
            
            return categorization
            
        except Exception as e:
            print(f"Error categorizing document {filename}: {e}")
            return {
                "document_type": "unknown",
                "primary_purpose": "unknown",
                "technical_level": "unknown",
                "target_audience": "general",
                "filename": filename,
                "error": str(e)
            }
