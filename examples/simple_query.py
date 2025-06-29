"""
Batch processing example: Process multiple documents efficiently.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from rag_manager import RAGManager
from config import Config

async def batch_processing_example():
    """Example of batch processing multiple documents."""
    
    print("🚀 Batch Processing Example")
    print("=" * 50)
    
    # Initialize RAG Manager
    rag_manager = RAGManager(use_existing_instance=True)
    await rag_manager.initialize()
    
    # Show initial status
    print("\n📊 Initial Status:")
    status = rag_manager.get_status()
    print(f"  - Processed files: {status['processed_files_count']}")
    print(f"  - Pending files: {status['pending_files_count']}")
    
    # Example 1: Process specific documents
    print("\n📄 Example 1: Processing specific documents")
    document_paths = [
        "path/to/document1.pdf",
        "path/to/document2.pdf", 
        "path/to/document3.docx"
    ]
    
    for doc_path in document_paths:
        if Path(doc_path).exists():
            print(f"Processing: {doc_path}")
            await rag_manager.process_document(doc_path)
        else:
            print(f"⚠️ File not found: {doc_path}")
    
    # Example 2: Process all pending documents
    print("\n📁 Example 2: Processing all pending documents")
    
    # First, let's add some demo files to pending directory
    pending_dir = Config.PENDING_DIR
    print(f"Pending directory: {pending_dir}")
    print("💡 To test batch processing:")
    print(f"  1. Copy your PDF files to: {pending_dir}")
    print("  2. Run this script again")
    
    # Check if there are pending files
    pending_files = list(pending_dir.glob("*.pdf"))
    if pending_files:
        print(f"Found {len(pending_files)} pending PDF files")
        results = await rag_manager.process_pending_documents()
        print(f"Processing results: {results}")
    else:
        print("No pending files found")
    
    # Example 3: Query after processing
    print("\n🤔 Example 3: Querying processed documents")
    
    try:
        # Ask comprehensive questions
        questions = [
            "What documents have been processed in this system?",
            "What are the common themes across all documents?",
            "Can you provide a summary of the main topics covered?",
            "Are there any numerical data, tables, or charts mentioned?",
            "What are the key findings from all the processed documents?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. Question: {question}")
            try:
                answer = await rag_manager.query(question, mode="hybrid")
                print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error during querying: {e}")
    
    # Final status
    print("\n📊 Final Status:")
    final_status = rag_manager.get_status()
    print(f"  - Processed files: {final_status['processed_files_count']}")
    print(f"  - Pending files: {final_status['pending_files_count']}")

async def demonstrate_file_management():
    """Demonstrate file management features."""
    print("\n🗂️ File Management Features:")
    print("=" * 40)
    
    # Show directory structure
    config = Config()
    
    print("📁 Directory Structure:")
    print(f"  - Documents: {config.DOCUMENTS_DIR}")
    print(f"  - Pending: {config.PENDING_DIR}")
    print(f"  - Processed: {config.PROCESSED_DIR}")
    print(f"  - Storage: {config.STORAGE_DIR}")
    print(f"  - LightRAG: {config.LIGHTRAG_STORAGE_DIR}")
    
    # Check existing files
    print("\n📄 File Inventory:")
    
    pending_files = list(config.PENDING_DIR.glob("*")) if config.PENDING_DIR.exists() else []
    processed_files = list(config.PROCESSED_DIR.glob("*")) if config.PROCESSED_DIR.exists() else []
    
    print(f"  - Pending files: {len(pending_files)}")
    for file in pending_files[:5]:  # Show first 5
        print(f"    • {file.name}")
    if len(pending_files) > 5:
        print(f"    ... and {len(pending_files) - 5} more")
    
    print(f"  - Processed files: {len(processed_files)}")
    for file in processed_files[:5]:  # Show first 5
        print(f"    • {file.name}")
    if len(processed_files) > 5:
        print(f"    ... and {len(processed_files) - 5} more")

async def main():
    """Main function."""
    await demonstrate_file_management()
    await batch_processing_example()

if __name__ == "__main__":
    asyncio.run(main())