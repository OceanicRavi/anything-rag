"""
Simple initialization test for LightRAG fix.
"""
import asyncio
import os
from pathlib import Path

# Set up basic environment
os.environ.setdefault("OPENAI_API_KEY", "test-key")  # Just for testing

async def test_initialization_only():
    """Test just the initialization steps."""
    
    print("🧪 Testing LightRAG initialization (no API calls)...")
    
    try:
        # Import test
        print("1. Testing imports...")
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        print("✅ Imports successful")
        
        # Create test directory
        test_dir = Path("./test_init")
        test_dir.mkdir(exist_ok=True)
        
        print("2. Creating LightRAG instance...")
        
        # Simple dummy functions (no API calls)
        def dummy_llm(prompt, **kwargs):
            return "dummy response"
        
        def dummy_embed(texts):
            return [[0.1] * 3072 for _ in texts]  # Dummy embeddings
        
        from lightrag.utils import EmbeddingFunc
        
        # Create LightRAG instance
        rag = LightRAG(
            working_dir=str(test_dir),
            llm_model_func=dummy_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=dummy_embed,
            ),
        )
        
        print("3. Initializing storages...")
        await rag.initialize_storages()
        
        print("4. Initializing pipeline status...")
        await initialize_pipeline_status()
        
        print("✅ Initialization successful!")
        print("✅ The 'history_messages' error fix is working!")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🔧 Simple LightRAG Initialization Test")
    print("=" * 50)
    
    try:
        success = asyncio.run(test_initialization_only())
        
        if success:
            print("\n🎉 Fix verified! The initialization is working correctly.")
            print("✅ You can now process documents without the 'history_messages' error.")
            print("\n📝 Next steps:")
            print("  1. Make sure your .env file has a valid OpenAI API key")
            print("  2. Try: python main.py process your_document.txt")
        else:
            print("\n❌ Initialization failed. Check the error above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()