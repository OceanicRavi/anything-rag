"""
Simple query interface for RAG-Anything.
"""
import asyncio
from typing import Optional
from rag_manager import RAGManager

class QueryInterface:
    """Simple interface for querying processed documents."""
    
    def __init__(self):
        self.rag_manager = None
    
    async def initialize(self, use_existing: bool = True):
        """Initialize the query interface."""
        self.rag_manager = RAGManager(use_existing_instance=use_existing)
        await self.rag_manager.initialize()
        print("✅ RAG system initialized successfully!")
    
    async def add_document(self, file_path: str, force_reprocess: bool = False):
        """Add and process a single document."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        success = await self.rag_manager.process_document(file_path, force_reprocess)
        if success:
            print(f"✅ Document processed successfully: {file_path}")
        else:
            print(f"❌ Failed to process document: {file_path}")
        return success
    
    async def process_all_pending(self, force_reprocess: bool = False):
        """Process all documents in the pending directory."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        results = await self.rag_manager.process_pending_documents(force_reprocess)
        print(f"📊 Processing results: {results}")
        return results
    
    async def ask(self, question: str, mode: str = "hybrid") -> str:
        """Ask a question and get an answer."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        try:
            answer = await self.rag_manager.query(question, mode)
            return answer
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    async def remove_document(self, doc_identifier: str) -> bool:
        """Remove a specific document and its knowledge context."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        try:
            print(f"🗑️ Removing document: {doc_identifier}")
            
            # First, list available documents to find exact match
            available_docs = self._get_processed_documents_list()
            
            # Find matching document
            matches = [doc for doc in available_docs if doc_identifier.lower() in doc.lower()]
            
            if not matches:
                print(f"❌ Document '{doc_identifier}' not found")
                print("Available documents:")
                for doc in available_docs:
                    print(f"  - {doc}")
                return False
            
            if len(matches) > 1:
                print(f"⚠️ Multiple matches found for '{doc_identifier}':")
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match}")
                
                choice = input("Enter number to select (or 'all' for all matches): ").strip()
                if choice.lower() == 'all':
                    selected_docs = matches
                else:
                    try:
                        idx = int(choice) - 1
                        selected_docs = [matches[idx]]
                    except (ValueError, IndexError):
                        print("❌ Invalid selection")
                        return False
            else:
                selected_docs = matches
            
            # Remove each selected document
            for doc_name in selected_docs:
                success = await self._remove_document_from_storage(doc_name)
                if success:
                    print(f"✅ Removed: {doc_name}")
                else:
                    print(f"❌ Failed to remove: {doc_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error removing document: {e}")
            return False
    
    async def remove_all_documents(self) -> bool:
        """Remove all processed documents and reset knowledge base."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        try:
            # Confirm with user
            docs = self._get_processed_documents_list()
            if not docs:
                print("ℹ️ No documents to remove")
                return True
            
            print(f"⚠️ This will remove {len(docs)} processed documents:")
            for doc in docs[:5]:  # Show first 5
                print(f"  - {doc}")
            if len(docs) > 5:
                print(f"  ... and {len(docs) - 5} more")
            
            confirm = input("\n❓ Are you sure? Type 'yes' to confirm: ").strip().lower()
            if confirm != 'yes':
                print("❌ Operation cancelled")
                return False
            
            # Remove all documents
            success_count = 0
            for doc_name in docs:
                if await self._remove_document_from_storage(doc_name):
                    success_count += 1
            
            # Clear storage directories
            await self._clear_storage_directories()
            
            print(f"✅ Removed {success_count}/{len(docs)} documents")
            print("✅ Knowledge base reset")
            
            return True
            
        except Exception as e:
            print(f"❌ Error removing all documents: {e}")
            return False
    
    async def list_processed_documents(self):
        """List all processed documents with details and source information."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        try:
            print("📋 Processed Documents")
            print("=" * 60)
            
            # Get documents with source information
            docs_info = self._get_detailed_documents_info()
            
            if not docs_info:
                print("ℹ️ No documents have been processed yet")
                print("\n💡 To add documents:")
                print("  - Copy files to ./documents/pending/")
                print("  - Run: python main.py process <file_path>")
                return
            
            # Display documents with details
            for i, (doc_name, info) in enumerate(docs_info.items(), 1):
                status_icons = []
                
                if info['in_cache']:
                    status_icons.append("💾")  # In cache
                if info['in_processed_dir']:
                    status_icons.append("📁")  # In processed directory
                if info['in_knowledge_base']:
                    status_icons.append("🧠")  # In knowledge base
                
                status = " ".join(status_icons) if status_icons else "❓"
                
                print(f"{i:2d}. {status} {doc_name}")
                
                # Show additional details if available
                if info['file_size']:
                    print(f"    📏 Size: {info['file_size']}")
                if info['process_date']:
                    print(f"    📅 Processed: {info['process_date']}")
            
            print(f"\n📊 Total: {len(docs_info)} unique documents")
            
            # Show legend
            print("\n🔍 Status Legend:")
            print("  💾 = In processing cache")
            print("  📁 = In processed files directory") 
            print("  🧠 = In knowledge base")
            
            # Show storage statistics
            status = self.rag_manager.get_status()
            print(f"\n📁 Storage Info:")
            print(f"  - Knowledge base: {status['storage_directories']['lightrag']}")
            print(f"  - Cache entries: {status['processed_files_count']}")
            print(f"  - Pending files: {status['pending_files_count']}")
            
            # Show knowledge base stats if available
            try:
                kb_stats = await self._get_knowledge_base_stats()
                if kb_stats:
                    print(f"\n🧠 Knowledge Base Stats:")
                    for key, value in kb_stats.items():
                        print(f"  - {key}: {value}")
            except:
                pass  # KB stats are optional
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
    
    def _get_detailed_documents_info(self) -> dict:
        """Get detailed information about processed documents."""
        try:
            from config import Config
            import os
            import json
            from datetime import datetime
            
            config = Config()
            docs_info = {}
            
            # Check cache
            cache_docs = set()
            if hasattr(self.rag_manager, '_processed_files_cache'):
                for cached_file in self.rag_manager._processed_files_cache:
                    if cached_file and isinstance(cached_file, str):
                        filename = os.path.basename(cached_file)
                        cache_docs.add(filename)
            
            # Check processed directory
            processed_docs = {}
            if config.PROCESSED_DIR.exists():
                for file_path in config.PROCESSED_DIR.iterdir():
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        try:
                            stat = file_path.stat()
                            processed_docs[file_path.name] = {
                                'size': f"{stat.st_size / 1024:.1f} KB",
                                'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                            }
                        except:
                            processed_docs[file_path.name] = {'size': 'Unknown', 'modified': 'Unknown'}
            
            # Check knowledge base
            kb_docs = set()
            try:
                full_docs_file = config.LIGHTRAG_STORAGE_DIR / "kv_store_full_docs.json"
                if full_docs_file.exists():
                    with open(full_docs_file, 'r', encoding='utf-8') as f:
                        docs_data = json.load(f)
                        for doc_id, doc_info in docs_data.items():
                            if isinstance(doc_info, dict) and 'file_path' in doc_info:
                                filename = os.path.basename(doc_info['file_path'])
                                kb_docs.add(filename)
            except:
                pass
            
            # Combine all document names
            all_doc_names = cache_docs | set(processed_docs.keys()) | kb_docs
            
            # Build detailed info for each document
            for doc_name in all_doc_names:
                if doc_name and len(doc_name) > 1 and not doc_name.startswith('.'):
                    docs_info[doc_name] = {
                        'in_cache': doc_name in cache_docs,
                        'in_processed_dir': doc_name in processed_docs,
                        'in_knowledge_base': doc_name in kb_docs,
                        'file_size': processed_docs.get(doc_name, {}).get('size'),
                        'process_date': processed_docs.get(doc_name, {}).get('modified')
                    }
            
            return docs_info
            
        except Exception as e:
            print(f"Warning: Could not get detailed document info: {e}")
            return {}
    
    async def _get_knowledge_base_stats(self) -> dict:
        """Get knowledge base statistics."""
        try:
            # Try to query the knowledge base for stats
            stats_query = "How many documents, entities, and relationships are in this knowledge base?"
            response = await self.ask(stats_query)
            
            # Parse basic numbers from response
            import re
            
            stats = {}
            
            # Look for numbers in the response
            doc_match = re.search(r'(\d+)\s*documents?', response, re.IGNORECASE)
            if doc_match:
                stats['Documents'] = doc_match.group(1)
            
            entity_match = re.search(r'(\d+)\s*entit(?:y|ies)', response, re.IGNORECASE)
            if entity_match:
                stats['Entities'] = entity_match.group(1)
            
            rel_match = re.search(r'(\d+)\s*relationships?', response, re.IGNORECASE)
            if rel_match:
                stats['Relationships'] = rel_match.group(1)
            
            return stats if stats else None
            
        except:
            return None
    
    async def clean_knowledge_base(self):
        """Interactive cleanup of knowledge base."""
        if not self.rag_manager:
            raise RuntimeError("Query interface not initialized. Call initialize() first.")
        
        print("🧹 Knowledge Base Cleanup")
        print("=" * 30)
        
        try:
            docs = self._get_processed_documents_list()
            
            if not docs:
                print("ℹ️ Knowledge base is already empty")
                return
            
            print(f"📊 Current knowledge base contains {len(docs)} documents")
            
            print("\n🔧 Cleanup options:")
            print("1. Remove specific document")
            print("2. Remove all documents")
            print("3. Show document details")
            print("4. Cancel")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                print("\nAvailable documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. {doc}")
                
                doc_choice = input("Enter document number or name: ").strip()
                try:
                    if doc_choice.isdigit():
                        doc_name = docs[int(doc_choice) - 1]
                    else:
                        doc_name = doc_choice
                    await self.remove_document(doc_name)
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
            
            elif choice == "2":
                await self.remove_all_documents()
            
            elif choice == "3":
                await self.list_processed_documents()
                
                # Show knowledge graph stats
                try:
                    answer = await self.ask("How many entities and relationships are in the knowledge base?")
                    print(f"\n🧠 Knowledge graph info:\n{answer}")
                except:
                    print("\n🧠 Could not retrieve knowledge graph statistics")
            
            elif choice == "4":
                print("❌ Cleanup cancelled")
            
            else:
                print("❌ Invalid option")
        
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
    
    def _get_processed_documents_list(self) -> list:
        """Get list of processed documents with proper deduplication."""
        try:
            from config import Config
            import os
            
            config = Config()
            all_docs = set()  # Use set for automatic deduplication
            
            # Get from cache (these are full file paths)
            if hasattr(self.rag_manager, '_processed_files_cache'):
                for cached_file in self.rag_manager._processed_files_cache:
                    if cached_file and isinstance(cached_file, str):
                        # Extract just the filename
                        filename = os.path.basename(cached_file)
                        if filename and filename != cached_file:  # Only if it was actually a path
                            all_docs.add(filename)
                        else:
                            all_docs.add(cached_file)  # It's already just a filename
            
            # Get from processed directory (these are already filenames)
            if config.PROCESSED_DIR.exists():
                for file_path in config.PROCESSED_DIR.iterdir():
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        all_docs.add(file_path.name)
            
            # Get from LightRAG storage (check what documents are actually in the knowledge base)
            try:
                if config.LIGHTRAG_STORAGE_DIR.exists():
                    # Check if there are any documents in the knowledge base
                    full_docs_file = config.LIGHTRAG_STORAGE_DIR / "kv_store_full_docs.json"
                    if full_docs_file.exists():
                        import json
                        with open(full_docs_file, 'r', encoding='utf-8') as f:
                            docs_data = json.load(f)
                            # Extract document names from the storage
                            for doc_id, doc_info in docs_data.items():
                                if isinstance(doc_info, dict) and 'file_path' in doc_info:
                                    filename = os.path.basename(doc_info['file_path'])
                                    all_docs.add(filename)
            except Exception as e:
                # If we can't read LightRAG storage, that's okay
                pass
            
            # Clean up and filter valid documents
            clean_docs = []
            for doc in all_docs:
                if doc and isinstance(doc, str):
                    # Remove any remaining path separators
                    clean_name = doc.replace('\\', '/').split('/')[-1]
                    # Filter out system files and empty names
                    if clean_name and not clean_name.startswith('.') and len(clean_name) > 1:
                        clean_docs.append(clean_name)
            
            # Remove exact duplicates and sort
            unique_docs = list(set(clean_docs))
            return sorted(unique_docs)
            
        except Exception as e:
            print(f"Warning: Could not get document list: {e}")
            return []
    
    async def _remove_document_from_storage(self, doc_name: str) -> bool:
        """Remove document from storage and cache."""
        try:
            # Remove from processed files cache
            cache_to_remove = []
            for cached_file in self.rag_manager._processed_files_cache:
                if doc_name in cached_file or cached_file.endswith(doc_name):
                    cache_to_remove.append(cached_file)
            
            for item in cache_to_remove:
                self.rag_manager._processed_files_cache.discard(item)
            
            # Save updated cache
            self.rag_manager._save_processed_files_cache()
            
            # Move file back to pending if it exists in processed
            from config import Config
            config = Config()
            
            processed_file = config.PROCESSED_DIR / doc_name
            if processed_file.exists():
                pending_file = config.PENDING_DIR / doc_name
                processed_file.rename(pending_file)
                print(f"📁 Moved {doc_name} back to pending directory")
            
            return True
            
        except Exception as e:
            print(f"Error removing document from storage: {e}")
            return False
    
    async def _clear_storage_directories(self):
        """Clear storage directories for complete reset."""
        try:
            from config import Config
            import shutil
            
            config = Config()
            
            # Clear LightRAG storage (this removes the knowledge graph)
            if config.LIGHTRAG_STORAGE_DIR.exists():
                shutil.rmtree(config.LIGHTRAG_STORAGE_DIR)
                config.LIGHTRAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                print("🗑️ Cleared knowledge graph storage")
            
            # Clear processed docs cache
            if config.CACHE_DIR.exists():
                cache_file = config.CACHE_DIR / "processed_files.json"
                if cache_file.exists():
                    cache_file.unlink()
                print("🗑️ Cleared document cache")
            
            # Move all processed files back to pending
            if config.PROCESSED_DIR.exists():
                for file in config.PROCESSED_DIR.iterdir():
                    if file.is_file():
                        (config.PENDING_DIR / file.name).write_bytes(file.read_bytes())
                        file.unlink()
                print("📁 Moved processed files back to pending")
            
        except Exception as e:
            print(f"Warning: Could not fully clear storage: {e}")
    
    def status(self):
        """Get system status."""
        if not self.rag_manager:
            print("❌ Query interface not initialized")
            return
        
        status = self.rag_manager.get_status()
        print("📊 System Status:")
        print(f"  - LightRAG storage exists: {status['lightrag_storage_exists']}")
        print(f"  - Processed files: {status['processed_files_count']}")
        print(f"  - Pending files: {status['pending_files_count']}")
        print(f"  - Storage directories:")
        for name, path in status['storage_directories'].items():
            print(f"    - {name}: {path}")
        return status

# Convenience functions for quick usage
async def quick_query(question: str, mode: str = "hybrid", use_existing: bool = True) -> str:
    """Quick query function for one-off questions."""
    interface = QueryInterface()
    await interface.initialize(use_existing=use_existing)
    return await interface.ask(question, mode)

async def quick_process_and_query(file_path: str, question: str, mode: str = "hybrid") -> str:
    """Quick function to process a document and ask a question."""
    interface = QueryInterface()
    await interface.initialize(use_existing=True)
    await interface.add_document(file_path)
    return await interface.ask(question, mode)

# Interactive CLI interface
async def interactive_mode():
    """Interactive command-line interface."""
    interface = QueryInterface()
    
    print("🚀 RAG-Anything Interactive Mode")
    print("Initializing...")
    
    await interface.initialize()
    interface.status()
    
    print("\n📚 Available commands:")
    print("  - ask <question>: Ask a question")
    print("  - add <file_path>: Add a document")
    print("  - process: Process all pending documents")
    print("  - status: Show system status")
    print("  - list: List processed documents")
    print("  - remove <doc_name>: Remove specific document")
    print("  - remove all: Remove all documents")
    print("  - clean: Interactive cleanup")
    print("  - quit: Exit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif command.lower() == 'status':
                interface.status()
            
            elif command.lower() == 'process':
                await interface.process_all_pending()
            
            elif command.lower() == 'list':
                await interface.list_processed_documents()
            
            elif command.lower() == 'clean':
                await interface.clean_knowledge_base()
            
            elif command.startswith('add '):
                file_path = command[4:].strip()
                await interface.add_document(file_path)
            
            elif command.startswith('remove '):
                target = command[7:].strip()
                if target.lower() == 'all':
                    await interface.remove_all_documents()
                else:
                    await interface.remove_document(target)
            
            elif command.startswith('ask '):
                question = command[4:].strip()
                if question:
                    print("🤔 Thinking...")
                    answer = await interface.ask(question)
                    print(f"\n💡 Answer:\n{answer}\n")
                else:
                    print("❌ Please provide a question")
            
            else:
                print("❌ Unknown command. Available commands:")
                print("  ask, add, process, status, list, remove, clean, quit")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_mode())