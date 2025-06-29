"""
Main application entry point for RAG-Anything.
"""
import asyncio
import sys
from pathlib import Path
from query_interface import QueryInterface, interactive_mode, quick_query, quick_process_and_query

async def main():
    """Main application function."""
    if len(sys.argv) < 2:
        print("🚀 RAG-Anything - Multimodal Document Query System")
        print("\nUsage:")
        print("  python main.py interactive              # Start interactive mode")
        print("  python main.py query <question>         # Quick query")
        print("  python main.py process <file_path>      # Process a document")
        print("  python main.py add-and-ask <file_path> <question>  # Process document and ask")
        print("  python main.py remove <doc_name>        # Remove specific document")
        print("  python main.py remove all               # Remove all documents")
        print("  python main.py list                     # List processed documents")
        print("  python main.py clean                    # Interactive cleanup")
        print("\nExamples:")
        print("  python main.py interactive")
        print("  python main.py query 'What are the main findings?'")
        print("  python main.py process ./documents/report.pdf")
        print("  python main.py add-and-ask ./report.pdf 'Summarize the key results'")
        print("  python main.py remove 'bank statement.pdf'")
        print("  python main.py remove all")
        print("  python main.py list")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "interactive":
            await interactive_mode()
        
        elif command == "query":
            if len(sys.argv) < 3:
                print("❌ Please provide a question")
                return
            
            question = " ".join(sys.argv[2:])
            print(f"🤔 Querying: {question}")
            answer = await quick_query(question)
            print(f"\n💡 Answer:\n{answer}")
        
        elif command == "process":
            if len(sys.argv) < 3:
                print("❌ Please provide a file path")
                return
            
            file_path = sys.argv[2]
            interface = QueryInterface()
            await interface.initialize()
            await interface.add_document(file_path)
        
        elif command == "add-and-ask":
            if len(sys.argv) < 4:
                print("❌ Please provide file path and question")
                return
            
            file_path = sys.argv[2]
            question = " ".join(sys.argv[3:])
            print(f"📄 Processing: {file_path}")
            print(f"🤔 Question: {question}")
            answer = await quick_process_and_query(file_path, question)
            print(f"\n💡 Answer:\n{answer}")
        
        elif command == "remove":
            if len(sys.argv) < 3:
                print("❌ Please provide a document name or 'all' to remove everything")
                return
            
            target = sys.argv[2]
            interface = QueryInterface()
            await interface.initialize()
            
            if target.lower() == "all":
                success = await interface.remove_all_documents()
            else:
                success = await interface.remove_document(target)
            
            if success:
                print(f"✅ Successfully removed: {target}")
            else:
                print(f"❌ Failed to remove: {target}")
        
        elif command == "list":
            interface = QueryInterface()
            await interface.initialize()
            await interface.list_processed_documents()
        
        elif command == "clean":
            interface = QueryInterface()
            await interface.initialize()
            await interface.clean_knowledge_base()
        
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python main.py' to see available commands")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())