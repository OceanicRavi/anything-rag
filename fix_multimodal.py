"""
Fix for multimodal processing issues.
"""

import asyncio
from config import Config

async def fix_and_test_multimodal():
    """Fix multimodal processing and test with the bank statement."""
    
    print("🔧 Fixing Multimodal Processing Issues...")
    print("=" * 50)
    
    # Check API key first
    config = Config()
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in the .env file first!")
        print("Edit .env file and add: OPENAI_API_KEY=sk-your-actual-key")
        return False
    
    try:
        # Initialize with proper multimodal support
        from raganything import RAGAnything
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        from lightrag.kg.shared_storage import initialize_pipeline_status
        
        print("1. Creating properly configured RAGAnything instance...")
        
        # Create LightRAG instance
        lightrag_instance = LightRAG(
            working_dir=str(config.LIGHTRAG_STORAGE_DIR),
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
                config.LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                **kwargs,
            ),
            embedding_func=EmbeddingFunc(
                embedding_dim=config.EMBEDDING_DIM,
                max_token_size=config.MAX_TOKEN_SIZE,
                func=lambda texts: openai_embed(
                    texts,
                    model=config.EMBEDDING_MODEL,
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL,
                ),
            ),
        )
        
        # Initialize storages
        await lightrag_instance.initialize_storages()
        await initialize_pipeline_status()
        
        print("2. Creating RAGAnything with complete multimodal support...")
        
        # Vision model function
        def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            if image_data:
                return openai_complete_if_cache(
                    config.VISION_MODEL,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]}
                    ],
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL,
                    **kwargs,
                )
            else:
                return openai_complete_if_cache(
                    config.LLM_MODEL,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL,
                    **kwargs,
                )
        
        # LLM model function for table processing
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                config.LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
                **kwargs,
            )
        
        # Create RAGAnything with ALL required functions
        rag_anything = RAGAnything(
            lightrag=lightrag_instance,
            vision_model_func=vision_model_func,
            llm_model_func=llm_model_func,  # This was missing!
            modal_caption_func=llm_model_func  # Add this explicitly
        )
        
        print("3. Testing basic query functionality...")
        
        # Test a simple query
        result = await rag_anything.query_with_multimodal(
            "What documents have been processed and what type of content do they contain?",
            mode="hybrid"
        )
        
        print(f"Query result: {result[:200]}...")
        
        print("\n✅ Multimodal processing fix appears to be working!")
        print("\n🎯 Now you can query your bank statement:")
        print("   - What financial transactions are shown in the bank statement?")
        print("   - What are the amounts and dates in the tables?")
        print("   - Can you summarize the financial data?")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fix: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_bank_statement_queries():
    """Test specific queries about the bank statement."""
    
    print("\n💰 Testing Bank Statement Queries...")
    print("=" * 40)
    
    try:
        from query_interface import QueryInterface
        
        interface = QueryInterface()
        await interface.initialize()
        
        bank_queries = [
            "What financial documents have been processed?",
            "What are the transaction amounts shown in the tables?",
            "Can you extract the dates and descriptions from the bank statement?",
            "What is the total of all transactions in the bank statement?",
            "What type of financial information is available in the processed documents?"
        ]
        
        for i, query in enumerate(bank_queries, 1):
            print(f"\n{i}. {query}")
            try:
                answer = await interface.ask(query, mode="hybrid")
                print(f"   💡 {answer[:300]}..." if len(answer) > 300 else f"   💡 {answer}")
            except Exception as e:
                print(f"   ❌ {e}")
        
        print("\n🎉 Bank statement analysis complete!")
        
    except Exception as e:
        print(f"❌ Bank statement query test failed: {e}")

async def main():
    """Main function."""
    success = await fix_and_test_multimodal()
    
    if success:
        await test_bank_statement_queries()
        
        print("\n🚀 Next Steps:")
        print("1. Use: python main.py interactive")
        print("2. Try queries like:")
        print("   'What transactions are in my bank statement?'")
        print("   'What are the amounts and dates in the financial data?'")
    else:
        print("\n❌ Fix incomplete. Check your API key and configuration.")

if __name__ == "__main__":
    asyncio.run(main())