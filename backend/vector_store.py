import os
import sys
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, ConnectionError
from langchain.docstore.document import Document

# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class VectorStore:
    """Handles vector storage and retrieval using OpenSearch"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = None
        self.index_name = Config.OPENSEARCH_INDEX
        self._connect_opensearch()
        self._create_index()
    
    def _connect_opensearch(self):
        """Connect to OpenSearch cluster"""
        try:
            self.client = OpenSearch(
                hosts=[{
                    'host': Config.OPENSEARCH_HOST,
                    'port': Config.OPENSEARCH_PORT
                }],
                http_auth=Config.OPENSEARCH_AUTH,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
            
            # Test connection
            if self.client.ping():
                print("Connected to OpenSearch")
            else:
                raise ConnectionError("Could not connect to OpenSearch")
                
        except Exception as e:
            print(f"Error connecting to OpenSearch: {e}")
            raise
    
    def _create_index(self):
        """Create index with proper mapping for hybrid search if it doesn't exist"""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index": {
                        "max_result_window": 10000
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "content_vector": {
                            "type": "knn_vector",
                            "dimension": Config.EMBEDDING_DIMENSION,
                            "method": {
                                "engine": "nmslib",
                                "space_type": "cosinesimil",
                                "name": "hnsw",
                                "parameters": {}
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "keyword"},
                                "document_id": {"type": "keyword"},
                                "chunk_id": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "total_chunks": {"type": "integer"},
                                "pages": {"type": "integer"},
                                "primary_page": {"type": "integer"},
                                "chunk_type": {"type": "keyword"},
                                "file_type": {"type": "keyword"}
                            }
                        }
                    }
                }
            }
            
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                print(f"Created index: {self.index_name}")
            except Exception as e:
                print(f"Error creating index: {e}")
                raise
    
    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """Generate embeddings for documents"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store"""
        if not documents:
            return []
        
        # Generate embeddings
        embeddings = self.embed_documents(documents)
        
        # Prepare documents for bulk indexing
        bulk_body = []
        document_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.metadata.get("chunk_id", f"doc_{i}")
            document_ids.append(doc_id)
            
            # Prepare document
            document = {
                "content": doc.page_content,
                "content_vector": embedding.tolist(),
                "metadata": doc.metadata
            }
            
            # Add to bulk body
            bulk_body.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            })
            bulk_body.append(document)
        
        # Bulk index
        try:
            response = self.client.bulk(body=bulk_body)
            
            # Check for errors
            if response.get('errors'):
                print("Some documents failed to index:")
                for item in response['items']:
                    if 'error' in item['index']:
                        print(f"Error: {item['index']['error']}")
            
            print(f"Indexed {len(documents)} documents")
            return document_ids
            
        except Exception as e:
            print(f"Error indexing documents: {e}")
            raise
    
    def hybrid_search(self, query: str, k: int = 5, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Document]:
        """Hybrid search combining vector similarity and keyword matching with configurable weights"""
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        vector_weight = vector_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # Hybrid search query
        search_body = {
            "size": k * 2,  # Get more results for better hybrid scoring
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "content_vector": {
                                    "vector": query_embedding.tolist(),
                                    "k": k * 2,
                                    "boost": vector_weight
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "content.keyword^3"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": keyword_weight
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Convert results to Documents with hybrid scores
            documents = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                metadata = source.get('metadata', {})
                
                # Combine vector and keyword scores
                vector_score = hit['_score'] if 'knn' in str(hit.get('_explanation', {})) else 0
                keyword_score = hit['_score'] - vector_score if vector_score > 0 else hit['_score']
                
                metadata.update({
                    "hybrid_score": hit['_score'],
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "search_type": "hybrid"
                })
                
                doc = Document(
                    page_content=source['content'],
                    metadata=metadata
                )
                documents.append(doc)
            
            # Sort by hybrid score and return top k
            documents.sort(key=lambda x: x.metadata.get('hybrid_score', 0), reverse=True)
            return documents[:k]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fallback to vector search
            return self.similarity_search(query, k)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search query
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": query_embedding.tolist(),
                        "k": k
                    }
                }
            }
        }
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Convert results to Documents
            documents = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                metadata = source.get('metadata', {})
                metadata['score'] = hit['_score']
                
                doc = Document(
                    page_content=source['content'],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        if not document_ids:
            return
        
        bulk_body = []
        for doc_id in document_ids:
            bulk_body.append({
                "delete": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            })
        
        try:
            response = self.client.bulk(body=bulk_body)
            print(f"Deleted {len(document_ids)} documents")
        except Exception as e:
            print(f"Error deleting documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            
            return {
                "index_name": self.index_name,
                "document_count": doc_count,
                "embedding_dimension": Config.EMBEDDING_DIMENSION,
                "embedding_model": Config.EMBEDDING_MODEL
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_index(self):
        """Clear all documents from index"""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"Deleted index: {self.index_name}")
                self._create_index()
        except Exception as e:
            print(f"Error clearing index: {e}")
