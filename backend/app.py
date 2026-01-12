import os
import sys
import json
import tempfile
import threading
import queue
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from typing import List, Dict, Any
import psutil

# Add config directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
from query_rewriter import QueryRewriter
from reranker import ChunkReranker
from document_summarizer import DocumentSummarizer
from observability import observability
from config.config import Config, validate_config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Configure app
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
pdf_processor = PDFProcessor(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP
)
vector_store = VectorStore()
rag_system = RAGSystem()
query_rewriter = QueryRewriter()
reranker = ChunkReranker()
document_summarizer = DocumentSummarizer()

# Global chat history (in production, use Redis or database)
chat_history = []

# Background processing queue
processing_queue = queue.Queue()
processing_status = {}
processing_lock = threading.Lock()

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """Validate file size against limits"""
    max_size = Config.MAX_CONTENT_LENGTH
    if file_size > max_size:
        return False
    
    # Additional memory-based validation
    available_memory = psutil.virtual_memory().available
    estimated_memory_usage = file_size * 3  # Rough estimate for processing
    
    if estimated_memory_usage > available_memory * 0.8:  # Don't use more than 80% of available memory
        return False
    
    return True

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(Config.UPLOAD_FOLDER)
    
    return {
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "disk_percent": disk.percent,
        "disk_available_gb": disk.free / (1024**3),
        "cpu_percent": psutil.cpu_percent(interval=1)
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Liveness probe: backend process is running"""
    return jsonify({
        "status": "ok",
        "message": "backend is running"
    }), 200

@app.route('/health/full', methods=['GET'])
def full_health_check():
    """Health check endpoint with observability"""
    try:
        # Test OpenSearch connection
        stats = vector_store.get_stats()
        
        # Get observability health status
        health_status = observability.get_health_status()
        
        # Merge both health checks
        merged_status = {
            "status": "healthy" if health_status['healthy'] else "degraded",
            "healthy": health_status['healthy'],
            "vector_store": stats,
            "observability": health_status,
            "config": {
                "embedding_model": Config.EMBEDDING_MODEL,
                "llm_model": Config.GEMINI_MODEL,
                "chunk_size": Config.CHUNK_SIZE
            }
        }
        
        status_code = 200 if health_status['healthy'] else 503
        return jsonify(merged_status), status_code
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "healthy": False,
            "error": str(e)
        }), 503


@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_files():
    """Upload and process PDF files with validation and background processing"""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        # Check system resources
        resources = get_system_resources()
        if resources['memory_percent'] > 90:
            return jsonify({"error": "System memory too high. Please try again later."}), 503
        
        # Validate file sizes
        total_size = 0
        saved_files = []

        for file in files:
            if file and file.filename:
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                total_size += file_size
                
                if not validate_file_size(file_size):
                    return jsonify({
                        "error": f"File {file.filename} is too large or system doesn't have enough memory",
                        "max_size": Config.MAX_CONTENT_LENGTH,
                        "file_size": file_size
                    }), 413

                # ✅ Save file NOW (inside request)
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                saved_files.append({
                    "filename": filename,
                    "path": file_path
                })

        # Generate batch ID
        batch_id = f"batch_{int(time.time())}_{len(saved_files)}"
        
        processing_data = {
            "batch_id": batch_id,
            "files": saved_files,   # ✅ paths only
            "total_files": len(saved_files),
            "total_size": total_size,
            "timestamp": time.time()
        }
        
        with processing_lock:
            processing_status[batch_id] = {
                "status": "queued",
                "progress": 0,
                "total_files": len(saved_files),
                "processed_files": 0,
                "message": "Files queued for processing",
                "start_time": time.time()
            }
        
        # Start background processing
        processing_queue.put(processing_data)
        
        return jsonify({
            "message": f"Files queued for processing",
            "batch_id": batch_id,
            "total_files": len(saved_files),
            "total_size": total_size,
            "estimated_time": max(30, len(saved_files) * 10)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def background_processor():
    """Background thread for processing PDFs"""
    while True:
        try:
            # Get processing task from queue
            processing_data = processing_queue.get(timeout=1)
            batch_id = processing_data["batch_id"]
            files = processing_data["files"]  # now: list of {filename, path}
            
            with processing_lock:
                processing_status[batch_id]["status"] = "processing"
                processing_status[batch_id]["message"] = "Starting file processing..."
            
            uploaded_files = []
            processed_documents = []
            errors = []
            
            for i, file_info in enumerate(files):
                filename = file_info["filename"]
                file_path = file_info["path"]

                if not allowed_file(filename):
                    continue

                # Update progress
                with processing_lock:
                    processing_status[batch_id]["processed_files"] = i
                    processing_status[batch_id]["progress"] = (i / len(files)) * 100
                    processing_status[batch_id]["message"] = f"Processing {filename}..."
                
                try:
                    # Process PDF (file already exists on disk)
                    documents = pdf_processor.process_pdf(file_path, filename)
                    processed_documents.extend(documents)
                    
                    # Get file info
                    file_info_meta = pdf_processor.get_pdf_info(file_path)
                    uploaded_files.append({
                        "filename": filename,
                        "file_path": file_path,
                        "chunks_created": len(documents),
                        "info": file_info_meta
                    })
                    
                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")

            # Add documents to vector store
            if processed_documents:
                with processing_lock:
                    processing_status[batch_id]["message"] = "Storing documents in vector database..."
                
                document_ids = vector_store.add_documents(processed_documents)
                
                # Generate document summaries
                with processing_lock:
                    processing_status[batch_id]["message"] = "Generating document summaries..."
                
                try:
                    # Group documents by document_id for summarization
                    doc_groups = {}
                    for doc in processed_documents:
                        doc_id = doc.metadata.get('document_id')
                        if doc_id:
                            doc_groups.setdefault(doc_id, []).append(doc)
                    
                    # Generate summaries for each unique document
                    summaries = []
                    for doc_id, doc_chunks in doc_groups.items():
                        combined_content = "\n\n".join(
                            [chunk.page_content for chunk in doc_chunks[:3]]
                        )
                        filename = doc_chunks[0].metadata.get('source', 'Unknown')
                        
                        summary = document_summarizer.summarize_document(
                            combined_content, filename
                        )
                        summary['document_id'] = doc_id
                        summaries.append(summary)
                    
                    # Attach summaries to uploaded files
                    for uf in uploaded_files:
                        uf['summary'] = next(
                            (s for s in summaries if s.get('filename') == uf.get('filename')),
                            None
                        )
                
                except Exception as e:
                    print(f"Error generating summaries: {e}")
                    # Continue without summaries
            else:
                document_ids = []
            
            # Update final status
            with processing_lock:
                processing_status[batch_id].update({
                    "status": "completed" if not errors else "completed_with_errors",
                    "progress": 100,
                    "processed_files": len(files),
                    "uploaded_files": uploaded_files,
                    "total_chunks": len(processed_documents),
                    "document_ids": document_ids,
                    "errors": errors,
                    "message": f"Completed processing {len(uploaded_files)} files",
                    "end_time": time.time()
                })
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Background processing error: {e}")
            if 'batch_id' in locals():
                with processing_lock:
                    processing_status[batch_id].update({
                        "status": "error",
                        "message": f"Processing failed: {str(e)}",
                        "end_time": time.time()
                    })


# Start background processor
background_thread = threading.Thread(target=background_processor, daemon=True)
background_thread.start()

@app.route('/processing-status/<batch_id>', methods=['GET'])
def get_processing_status(batch_id):
    """Get processing status for a batch"""
    with processing_lock:
        if batch_id not in processing_status:
            return jsonify({"error": "Batch ID not found"}), 404
        
        status = processing_status[batch_id].copy()
        
        # Add estimated time remaining
        if status["status"] == "processing" and status["processed_files"] > 0:
            avg_time_per_file = (time.time() - status["start_time"]) / status["processed_files"]
            remaining_files = status["total_files"] - status["processed_files"]
            status["estimated_remaining_time"] = int(avg_time_per_file * remaining_files)
        
        return jsonify(status)

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    """Chat with the RAG system"""
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Rewrite query for better retrieval
        rewritten_question = query_rewriter.rewrite_query(question)
        
        # Two-stage retrieval: First stage - get more documents
        initial_docs = vector_store.hybrid_search(
            rewritten_question, 
            k=Config.TOP_K_RETRIEVAL * 3  # Get 3x more for re-ranking
        )
        
        # Second stage - re-rank using LLM
        context_docs = reranker.rerank_documents(
            question, 
            initial_docs, 
            top_k=Config.TOP_K_RETRIEVAL
        )
        #context_docs = initial_docs[:Config.TOP_K_RETRIEVAL]
        
        # Filter by confidence threshold if needed
        if Config.SIMILARITY_THRESHOLD > 0:
            context_docs = [
                doc for doc in context_docs 
                if doc.metadata.get('hybrid_score', doc.metadata.get('score', 0)) >= Config.SIMILARITY_THRESHOLD
            ]
        
        # Generate response
        response = rag_system.chat_with_history(
            question,  # Use original question for response
            context_docs, 
            chat_history
        )
        
        # Add query rewriting and re-ranking info
        response['rewritten_query'] = rewritten_question if rewritten_question != question else None
        response['query_expansion_used'] = rewritten_question != question
        response['reranking_used'] = len(initial_docs) > Config.TOP_K_RETRIEVAL
        response['initial_docs_retrieved'] = len(initial_docs)
        response['final_docs_after_reranking'] = len(context_docs)
        
        # Record observability metrics
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        observability.record_request({
            'latency': latency,
            'confidence_score': response.get('confidence_score', 0),
            'document_count': len(context_docs),
            'search_type': 'hybrid_reranked',
            'not_found': response.get('not_found', False),
            'error_type': None
        })
        
        # Add to chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response["answer"]})
        
        # Keep history manageable
        if len(chat_history) > 20:
            chat_history[:] = chat_history[-20:]
        
        return jsonify(response)
        
    except Exception as e:
        # Record error metrics
        latency = (time.time() - start_time) * 1000
        observability.record_request({
            'latency': latency,
            'confidence_score': 0,
            'document_count': 0,
            'search_type': 'error',
            'not_found': False,
            'error_type': str(e)
        })
        
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_data():
    """Clear all data from vector store"""
    try:
        vector_store.clear_index()
        chat_history.clear()
        return jsonify({"message": "All data cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        vector_stats = vector_store.get_stats()
        observability_stats = observability.get_current_stats()
        health_status = observability.get_health_status()
        
        return jsonify({
            "vector_store": vector_stats,
            "chat_history_length": len(chat_history),
            "config": {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "top_k": Config.TOP_K_RETRIEVAL,
                "similarity_threshold": Config.SIMILARITY_THRESHOLD
            },
            "observability": observability_stats,
            "health": health_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get detailed observability metrics"""
    try:
        return jsonify(observability.get_current_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    """List uploaded files"""
    try:
        files = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_info = pdf_processor.get_pdf_info(file_path)
                    files.append(file_info)
        
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download uploaded file"""
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename,
            as_attachment=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Validate configuration
    try:
        validate_config()
        print("Configuration validated successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        exit(1)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=Config.DEBUG
    )
