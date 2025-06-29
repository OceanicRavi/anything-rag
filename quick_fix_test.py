"""
Quick test script to verify the LightRAG initialization fix.
"""
import asyncio
import os
from pathlib import Path

# Set up basic environment
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

async def test_lightrag_initialization():
    """Test proper LightRAG initialization."""
    
    print("🧪 Testing LightRAG initialization fix...")
    
    try:
        # Create test directory
        test_dir = Path("./test_storage")
        test_dir.mkdir(exist_ok=True)
        
        print("1. Creating LightRAG instance...")
        
        # Create LightRAG instance
        rag = LightRAG(
            working_dir=str(test_dir),
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("OPENAI_API_KEY"),
                **kwargs,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=os.getenv("OPENAI_API_KEY"),
                ),
            ),
        )
        
        print("2. Initializing storages...")
        await rag.initialize_storages()
        
        print("3. Initializing pipeline status...")
        await initialize_pipeline_status()
        
        print("4. Testing text insertion...")
        # Test with a simple text
        test_text = "This is a test document to verify LightRAG is working correctly."
        await rag.ainsert(test_text)  # 👈 Changed from insert() to ainsert()
        
        print("✅ LightRAG initialization successful!")
        print("✅ The 'history_messages' error should now be fixed!")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

async def main():
    """Main test function."""
    print("🔧 Quick Fix Test for LightRAG 'history_messages' Error")
    print("=" * 60)
    
    success = await test_lightrag_initialization()
    
    if success:
        print("\n🎉 Fix verified! You can now run:")
        print("   python main.py process your_document.txt")
    else:
        print("\n❌ Fix needs more work. Check your API key and dependencies.")

if __name__ == "__main__":
    asyncio.run(main())