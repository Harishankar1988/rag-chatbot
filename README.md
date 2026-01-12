# PDF Chatbot RAG System

A comprehensive **GenAI RAG project** that allows users to upload PDF documents and chat with them using **Retrieval-Augmented Generation (RAG)** powered by **LLaMA running locally via Ollama** and **OpenSearch** as the vector database.

This project is designed to be **offline-first**, **cost-free for inference**, and suitable for learning, experimentation, and future production hardening.

## Features

- **PDF Upload & Processing**: Upload multiple PDF files with automatic text extraction and chunking
- **Vector Storage**: Store document embeddings in OpenSearch for efficient similarity search
- **RAG System**: Retrieve relevant document chunks and generate answers using LLaMA LLM via Ollama
- **Query Enhancement**: Automatic query rewriting and reranking for better retrieval accuracy
- **Document Summarization**: Generate document-level summaries and extract key topics
- **Observability**: Built-in monitoring and performance tracking
- **Modern UI**: Beautiful Streamlit frontend with chat interface
- **Real-time Chat**: Conversational AI with context awareness and chat history
- **Source Citation**: Answers include source document references with page numbers

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Flask API     │    │  OpenSearch     │
│   Frontend      │◄──►│   Backend       │◄──►│  Vector DB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  LLaMA LLM      │
                       │  (Ollama Local) │
                       └─────────────────┘
```

## Tech Stack

- **Frontend**: Streamlit with streamlit-chat
- **Backend**: Flask with Flask-CORS and Flask-Limiter
- **PDF Processing**: PyPDF2, pdfplumber, LangChain
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: OpenSearch with k-NN plugin
- **LLM**: LLaMA 3 via Ollama (local inference)
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Query Enhancement**: Cross-encoder reranking (MS-MARCO-MiniLM)
- **Monitoring**: Custom observability module with system metrics

## Prerequisites

1. **Python 3.10**
2. **OpenSearch Server** with k-NN plugin
3. **Ollama** with LLaMA 3 model installed
4. **Docker** (for OpenSearch container)

### OpenSearch Setup

You can quickly set up OpenSearch using Docker:

```bash
# Create docker-compose.yml
cat > docker-compose.yml << EOF
version: '3'
services:
  opensearch:
    image: opensearchproject/opensearch:2.11.0
    container_name: opensearch
    environment:
      - discovery.type=single-node
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin123
      - plugins.security.disabled=true
    ports:
      - "9200:9200"
      - "9600:9600"
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    networks:
      - opensearch-net

volumes:
  opensearch-data:

networks:
  opensearch-net:
EOF

# Start OpenSearch
docker-compose up -d

# Wait for it to be ready (check with curl)
curl -X GET "localhost:9200" -u admin:admin123
```

### Ollama Setup

Install and set up Ollama for local LLaMA inference:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull LLaMA 3 model (recommended)
ollama pull llama3

# Verify installation
ollama list
```

**Available Models:**
- `llama3` (recommended) - 8B parameters, good balance of speed and quality
- `llama3:70b` - Higher quality but requires more resources
- `codellama` - Specialized for code-related queries

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd pdf-chatbot-rag
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Configure your .env file**:
```env
# Flask Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=pdf_documents
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin123

# Ollama Configuration
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Text Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.7

# File Upload Configuration
MAX_CONTENT_LENGTH=33554432  # 32MB in bytes
```

## Usage

### Quick Start (Recommended)

Use the provided startup script for automatic setup:

```bash
# Make script executable
chmod +x start.sh

# Start the complete system
./start.sh

# Development mode (backend only)
./start.sh --dev
```

The script will:
- Check and install dependencies
- Verify Ollama is running
- Start OpenSearch container
- Launch backend and frontend services

### Manual Start

#### 1. Start the Backend Server

```bash
cd backend
python app.py
```

The backend will start on `http://localhost:5001`

#### 2. Start the Frontend

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

The frontend will open in your browser at `http://localhost:8501`

### 3. Use the Application

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Wait for Processing**: The system will extract text, create embeddings, and store them
3. **Chat**: Ask questions about your uploaded documents
4. **View Sources**: Each answer includes source document references

## API Endpoints

### Backend API (Flask)

- `GET /health` - Health check and system stats
- `POST /upload` - Upload and process PDF files
- `POST /chat` - Chat with the RAG system (with query enhancement and reranking)
- `POST /clear` - Clear all data
- `GET /stats` - Get system statistics and resource usage
- `GET /files` - List uploaded files
- `GET /summarize/<filename>` - Get document summary and key topics
- `GET /observability/metrics` - Get performance and system metrics

### Example API Usage

```bash
# Health check
curl http://localhost:5001/health

# Upload a PDF
curl -X POST -F "files=@document.pdf" http://localhost:5001/upload

# Chat
curl -X POST -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}' \
     http://localhost:5001/chat

# Get document summary
curl http://localhost:5001/summarize/document.pdf

# Get system metrics
curl http://localhost:5001/observability/metrics
```

## Configuration

### Text Processing Parameters

- `CHUNK_SIZE`: Number of characters per chunk (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)

### RAG Parameters

- `TOP_K_RETRIEVAL`: Number of documents to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)
- `OLLAMA_MODEL`: Ollama model to use (default: llama3)
- `OLLAMA_URL`: Ollama API endpoint (default: http://localhost:11434/api/generate)

### Advanced Features

- **Query Rewriting**: Automatically enhances user queries for better retrieval
- **Cross-Encoder Reranking**: Uses MS-MARCO-MiniLM for improved result ranking
- **Document Summarization**: Generates summaries and extracts key topics
- **Observability**: Tracks performance metrics and system resource usage

## Project Structure

```
pdf-chatbot-rag/
├── backend/
│   ├── app.py                  # Flask API server with enhanced endpoints
│   ├── pdf_processor.py        # PDF processing and chunking
│   ├── vector_store.py         # OpenSearch integration
│   ├── rag_system.py           # RAG system with LLaMA/Ollama
│   ├── rag_system_2.py         # Alternative RAG implementation
│   ├── ollama_client.py        # Ollama API client
│   ├── query_rewriter.py       # Query enhancement module
│   ├── reranker.py             # Cross-encoder reranking
│   ├── document_summarizer.py  # Document summarization
│   └── observability.py        # Monitoring and metrics
├── frontend/
│   └── app.py                  # Streamlit frontend
├── config/
│   └── config.py               # Configuration management
├── data/
│   └── uploaded/               # Uploaded PDF files
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── docker-compose.yml          # OpenSearch container setup
├── start.sh                   # Startup script for easy deployment
└── README.md                  # This file
```

## Troubleshooting

### Common Issues

1. **OpenSearch Connection Error**:
   - Ensure OpenSearch is running
   - Check host/port configuration
   - Verify credentials

2. **Ollama Connection Error**:
   - Ensure Ollama service is running (`ollama serve`)
   - Check if LLaMA model is installed (`ollama list`)
   - Verify Ollama URL (default: http://localhost:11434)

3. **PDF Processing Error**:
   - Ensure PDF files are not corrupted
   - Check file size limits (32MB default)
   - Verify sufficient memory for processing

4. **Embedding Issues**:
   - Ensure sufficient RAM for embedding models
   - Check internet connection for model downloads
   - Verify embedding model configuration

5. **Performance Issues**:
   - Monitor system resources via `/observability/metrics`
   - Adjust `TOP_K_RETRIEVAL` and chunking parameters
   - Consider using smaller LLaMA models for resource-constrained systems

### Debug Mode

Enable debug mode in `.env`:
```env
DEBUG=True
```

This will provide detailed error messages and auto-reload the Flask server.

## Performance Optimization

1. **Vector Search**: Use appropriate `TOP_K_RETRIEVAL` values (3-10 recommended)
2. **Chunking**: Adjust `CHUNK_SIZE` based on document types (500-1500 for most docs)
3. **Embeddings**: Consider using larger models for better accuracy
4. **Reranking**: Cross-encoder reranking improves precision but adds latency
5. **Model Selection**: 
   - `llama3` for balanced performance
   - `llama3:70b` for higher quality (requires more resources)
6. **System Resources**: Monitor via observability endpoints and adjust accordingly
7. **Caching**: Enable response caching for frequently asked questions

## Future Scope & Roadmap

### 🚧 **In Development**

#### **Snowflake Cortex Integration**
- **Hybrid Data Sources**: Combine vector store retrieval with Snowflake Cortex for structured/unstructured data
- **Unified RAG Pipeline**: Seamlessly blend document chunks with database records
- **Advanced Analytics**: Leverage Snowflake's ML capabilities for enhanced insights
- **Data Governance**: Implement proper access controls and data lineage

#### **Intelligent Caching System**
- **Multi-Level Caching**: 
  - L1: In-memory cache for frequent queries
  - L2: Redis cache for session persistence
  - L3: Persistent cache for common document patterns
- **Smart Cache Invalidation**: Automatic cache updates when documents are modified
- **Semantic Caching**: Cache based on query similarity, not exact matches
- **Cache Analytics**: Monitor cache hit rates and performance improvements

#### **MLOps Pipeline**
- **Model Versioning**: Track and manage different LLaMA model versions
- **A/B Testing**: Compare different retrieval and generation strategies
- **Performance Monitoring**: Real-time metrics for latency, accuracy, and user satisfaction
- **Automated Retraining**: Schedule embedding model updates and fine-tuning
- **Model Registry**: Centralized repository for all ML artifacts

#### **Comprehensive Testing Suite**
- **Unit Tests**: Individual component testing with >90% coverage
- **Integration Tests**: End-to-end API and database testing
- **Performance Tests**: Load testing and stress testing scenarios
- **LLM Evaluation**: Automated assessment of response quality and relevance
- **Security Tests**: Vulnerability scanning and penetration testing

### 📋 **Planned Enhancements**

#### **Advanced RAG Features**
- **Multi-Modal Support**: Handle images, tables, and charts in PDFs
- **Hierarchical Chunking**: Document structure-aware segmentation
- **Knowledge Graphs**: Entity relationships and concept mapping
- **Conversation Memory**: Long-term context retention across sessions

#### **User Experience**
- **Real-time Collaboration**: Multi-user document analysis
- **Export Features**: Generate reports and summaries in multiple formats
- **Mobile Support**: Responsive design for mobile devices
- **Accessibility**: WCAG compliance and screen reader support

#### **Enterprise Features**
- **SSO Integration**: LDAP, OAuth, and SAML authentication
- **Role-Based Access Control**: Granular permissions and user management
- **Audit Logging**: Comprehensive activity tracking and compliance
- **Multi-tenant Architecture**: Isolated environments for different organizations

### 🎯 **Technical Goals**

#### **Performance Targets**
- **Response Time**: <2 seconds for typical queries
- **Throughput**: 100+ concurrent users
- **Accuracy**: >85% relevance score on benchmark datasets
- **Uptime**: 99.9% availability with automatic failover

#### **Scalability**
- **Horizontal Scaling**: Load balancing across multiple instances
- **Database Optimization**: Efficient indexing and query optimization
- **Resource Management**: Dynamic resource allocation based on load
- **CDN Integration**: Global content delivery for faster responses

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for detailed error messages
3. Create an issue in the repository

---

**Note**: This system is designed for research and development purposes. For production use, consider additional security measures, monitoring, and scalability improvements.
