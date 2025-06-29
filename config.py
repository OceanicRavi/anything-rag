"""
Configuration management for RAG-Anything project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG-Anything project."""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Model Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Storage Configuration
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
    LIGHTRAG_STORAGE_DIR = STORAGE_DIR / "lightrag_storage"
    PROCESSED_DOCS_DIR = STORAGE_DIR / "processed_docs"
    CACHE_DIR = STORAGE_DIR / "cache"
    
    # Document Processing Configuration
    DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
    PENDING_DIR = DOCUMENTS_DIR / "pending"
    PROCESSED_DIR = DOCUMENTS_DIR / "processed"
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
    
    # Processing Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    PARSE_METHOD = os.getenv("PARSE_METHOD", "auto")
    DEVICE = os.getenv("DEVICE", "cpu")
    
    # Embedding Configuration
    EMBEDDING_DIM = 3072  # text-embedding-3-large dimension
    MAX_TOKEN_SIZE = 8192
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and create necessary directories."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        
        # Create necessary directories
        for directory in [
            cls.STORAGE_DIR,
            cls.LIGHTRAG_STORAGE_DIR,
            cls.PROCESSED_DOCS_DIR,
            cls.CACHE_DIR,
            cls.DOCUMENTS_DIR,
            cls.PENDING_DIR,
            cls.PROCESSED_DIR,
            cls.LOG_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_processed_files_list(cls):
        """Get list of already processed files."""
        processed_files = []
        if cls.PROCESSED_DIR.exists():
            for file_path in cls.PROCESSED_DIR.rglob("*"):
                if file_path.is_file():
                    processed_files.append(file_path.name)
        return processed_files
    
    @classmethod
    def mark_file_as_processed(cls, file_path: Path):
        """Mark a file as processed by moving it to processed directory."""
        if file_path.exists() and file_path.parent == cls.PENDING_DIR:
            destination = cls.PROCESSED_DIR / file_path.name
            file_path.rename(destination)
            return destination
        return file_path