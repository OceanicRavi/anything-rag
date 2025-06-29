"""
FastAPI server for RAG-Anything system.
Provides REST API endpoints for all document processing and querying functionality.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import shutil

from query_interface import QueryInterface
from rag_manager import RAGManager
from config import Config

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Anything API",
    description="REST API for multimodal document processing and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global interface instance
query_interface: Optional[QueryInterface] = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"

class QueryResponse(BaseModel):
    answer: str
    mode: str
    success: bool

class ProcessRequest(BaseModel):
    file_path: str
    force_reprocess: bool = False

class ProcessResponse(BaseModel):
    success: bool
    message: str
    file_path: str

class RemoveRequest(BaseModel):
    document_name: str
    remove_all: bool = False

class RemoveResponse(BaseModel):
    success: bool
    message: str
    removed_documents: List[str]

class DocumentInfo(BaseModel):
    name: str
    in_cache: bool
    in_processed_dir: bool
    in_knowledge_base: bool
    file_size: Optional[str]
    process_date: Optional[str]

class StatusResponse(BaseModel):
    lightrag_storage_exists: bool
    processed_files_count: int
    pending_files_count: int
    storage_directories: Dict[str, str]
    documents: List[DocumentInfo]

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the query interface on startup."""
    global query_interface
    try:
        logger.info("Initializing RAG-Anything API...")
        query_interface = QueryInterface()
        await query_interface.initialize(use_existing=True)
        logger.info("✅ RAG-Anything API initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG-Anything API...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {"status": "healthy", "service": "RAG-Anything API"}

# Query endpoints
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the processed documents."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        answer = await query_interface.ask(request.question, mode=request.mode)
        
        return QueryResponse(
            answer=answer,
            mode=request.mode,
            success=True
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/query/{question}")
async def quick_query(question: str, mode: str = "hybrid"):
    """Quick query via GET request."""
    request = QueryRequest(question=question, mode=mode)
    return await query_documents(request)

# Document processing endpoints
@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process a document from file path."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    try:
        # Process in background for large files
        background_tasks.add_task(
            _process_document_background,
            str(file_path),
            request.force_reprocess
        )
        
        return ProcessResponse(
            success=True,
            message=f"Document processing started for {file_path.name}",
            file_path=str(file_path)
        )
    except Exception as e:
        logger.error(f"Process request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    force_reprocess: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a document."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt', '.md'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
        )
    
    # Save uploaded file to pending directory
    config = Config()
    config.validate_config()
    
    try:
        pending_file = config.PENDING_DIR / file.filename
        
        # Save uploaded content
        with open(pending_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        background_tasks.add_task(
            _process_document_background,
            str(pending_file),
            force_reprocess
        )
        
        return {
            "success": True,
            "message": f"File uploaded and processing started: {file.filename}",
            "filename": file.filename,
            "file_path": str(pending_file)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/pending")
async def process_all_pending(background_tasks: BackgroundTasks):
    """Process all documents in the pending directory."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        background_tasks.add_task(_process_all_pending_background)
        
        return {
            "success": True,
            "message": "Processing all pending documents started"
        }
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Document management endpoints
@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all processed documents."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        docs_info = query_interface._get_detailed_documents_info()
        
        documents = []
        for doc_name, info in docs_info.items():
            documents.append(DocumentInfo(
                name=doc_name,
                in_cache=info['in_cache'],
                in_processed_dir=info['in_processed_dir'],
                in_knowledge_base=info['in_knowledge_base'],
                file_size=info['file_size'],
                process_date=info['process_date']
            ))
        
        return documents
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_name}")
async def remove_document(document_name: str):
    """Remove a specific document."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        success = await query_interface.remove_document(document_name)
        
        if success:
            return {
                "success": True,
                "message": f"Document '{document_name}' removed successfully",
                "removed_documents": [document_name]
            }
        else:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove document failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")

@app.delete("/documents")
async def remove_all_documents():
    """Remove all processed documents."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        success = await query_interface.remove_all_documents()
        
        if success:
            return {
                "success": True,
                "message": "All documents removed successfully",
                "removed_documents": ["all"]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to remove all documents")
            
    except Exception as e:
        logger.error(f"Remove all documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove all documents: {str(e)}")

# Status and information endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and document information."""
    if query_interface is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        status = query_interface.rag_manager.get_status()
        docs_info = query_interface._get_detailed_documents_info()
        
        documents = []
        for doc_name, info in docs_info.items():
            documents.append(DocumentInfo(
                name=doc_name,
                in_cache=info['in_cache'],
                in_processed_dir=info['in_processed_dir'],
                in_knowledge_base=info['in_knowledge_base'],
                file_size=info['file_size'],
                process_date=info['process_date']
            ))
        
        return StatusResponse(
            lightrag_storage_exists=status['lightrag_storage_exists'],
            processed_files_count=status['processed_files_count'],
            pending_files_count=status['pending_files_count'],
            storage_directories=status['storage_directories'],
            documents=documents
        )
    except Exception as e:
        logger.error(f"Get status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "RAG-Anything API",
        "version": "1.0.0",
        "description": "REST API for multimodal document processing and querying",
        "endpoints": {
            "health": "GET /health - Health check",
            "query": "POST /query - Query documents",
            "quick_query": "GET /query/{question} - Quick query",
            "upload": "POST /upload - Upload and process document",
            "process": "POST /process - Process document from file path",
            "process_pending": "POST /process/pending - Process all pending documents",
            "list_documents": "GET /documents - List processed documents",
            "remove_document": "DELETE /documents/{name} - Remove specific document",
            "remove_all": "DELETE /documents - Remove all documents",
            "status": "GET /status - Get system status"
        },
        "docs": "/docs - Interactive API documentation"
    }

# Background task functions
async def _process_document_background(file_path: str, force_reprocess: bool):
    """Background task for document processing."""
    try:
        logger.info(f"Background processing: {file_path}")
        success = await query_interface.add_document(file_path, force_reprocess)
        if success:
            logger.info(f"✅ Successfully processed: {file_path}")
        else:
            logger.error(f"❌ Failed to process: {file_path}")
    except Exception as e:
        logger.error(f"Background processing error: {e}")

async def _process_all_pending_background():
    """Background task for processing all pending documents."""
    try:
        logger.info("Background processing: all pending documents")
        results = await query_interface.process_all_pending()
        logger.info(f"✅ Batch processing complete: {results}")
    except Exception as e:
        logger.error(f"Background batch processing error: {e}")

# Run the server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG-Anything API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("🚀 Starting RAG-Anything API Server...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 Interactive docs at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )