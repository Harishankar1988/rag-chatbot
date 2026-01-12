import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the PDF Chatbot RAG system"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'uploaded')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # OpenSearch Configuration
    OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
    OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT', 9200))
    OPENSEARCH_INDEX = os.getenv('OPENSEARCH_INDEX', 'pdf_documents')
    OPENSEARCH_AUTH = (
        os.getenv('OPENSEARCH_USERNAME', 'admin'),
        os.getenv('OPENSEARCH_PASSWORD', 'admin')
    )
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384
    
    # Text Chunking Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    
    # Gemini LLM Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    # RAG Configuration
    TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', 5))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))

def validate_config():
    """Validate required configuration parameters"""
    if not Config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    return True
