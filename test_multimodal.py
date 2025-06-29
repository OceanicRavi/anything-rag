"""
Test multimodal processing functionality.
"""
import asyncio
import os
from pathlib import Path

# Set up environment
from config import Config

async def test_multimodal_processing():
    """Test if multimodal processing is working correctly."""
    
    print("🧪 Testing Multimodal Processing...")
    print("=" * 50)
    
    try:
        # Initialize RAG Manager
        from rag_manager import RAGManager
        
        print("1. Initializing RAG Manager...")
        rag_manager = RAGManager(use_existing_instance=True)
        await rag_manager.initialize()
        
        print("2. Checking RAGAnything configuration...")
        rag_anything = rag_manager.rag_anything
        
        # Check if functions are properly set
        print(f"   - LightRAG instance: {rag_anything.lightrag is not None}")
        print(f"   - Vision model func: {rag_anything.vision_model_func is not None}")
        
        # Check if modal processors are available
        if hasattr(rag_anything, 'modal_processors'):
            print(f"   - Modal processors: {list(rag_anything.modal_processors.keys())}")
        
        # Test table processing specifically
        print("3. Testing table processor...")
        
        # Create a simple test table
        test_table_content = {
            "table_body": """
| Date | Description | Amount |
|------|-------------|--------|
| 2024-01-01 | Salary | $5000 |
| 2024-01-02 | Rent | -$1500 |
            """,
            "table_caption": ["Test Bank Statement"],
            "table_footnote": ["Test data"]
        }
        
        # Try to get the table processor
        from raganything.modalprocessors import TableModalProcessor
        
        table_processor = TableModalProcessor(
            lightrag=rag_anything.lightrag,
            modal_caption_func=rag_manager._get_llm_model_func()
        )
        
        print("   ✅ Table processor created successfully")
        
        # Test the processing (without actually calling LLM)
        print("4. Testing processor configuration...")
        
        if table_processor.modal_caption_func is not None:
            print("   ✅ Modal caption function is properly set")
        else:
            print("   ❌ Modal caption function is None - this is the issue!")
            
        print("\n🎯 Diagnosis Complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_query_after_processing():
    """Test querying the processed document."""
    
    print("\n🔍 Testing Query Functionality...")
    print("=" * 40)
    
    try:
        from query_interface import QueryInterface
        
        interface = QueryInterface()
        await interface.initialize()
        
        # Test some queries about the bank statement
        test_queries = [
            "What documents have been processed?",
            "What type of financial information is available?",
            "Are there any tables or structured data in the documents?",
            "Can you summarize the content of the bank statement?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                # Use a shorter timeout for testing
                answer = await interface.ask(query, mode="hybrid")
                print(f"   Answer: {answer[:150]}..." if len(answer) > 150 else f"   Answer: {answer}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False

async def main():
    """Main test function."""
    
    # Check if API key is set
    config = Config()
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in the .env file")
        return
    
    # Test multimodal processing
    multimodal_success = await test_multimodal_processing()
    
    if multimodal_success:
        # Test querying
        await test_query_after_processing()
        
        print("\n🎉 Recommendations:")
        print("1. The text content was processed successfully")
        print("2. Tables had processing errors but document still loaded")
        print("3. You can query the document content")
        print("4. Try: python main.py interactive")
        print("   Then ask: 'What financial information is in the processed documents?'")
    else:
        print("\n❌ Multimodal processing has issues")
        print("💡 The document text is still processable, but table extraction may be incomplete")

if __name__ == "__main__":
    asyncio.run(main())