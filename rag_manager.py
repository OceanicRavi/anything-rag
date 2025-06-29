"""
RAG Manager: Core class for managing RAG-Anything operations.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from config import Config

class RAGManager:
    """Main RAG manager for document processing and querying."""
    
    def __init__(self, use_existing_instance: bool = True):
        """
        Initialize RAG Manager.
        
        Args:
            use_existing_instance: Whether to load existing LightRAG instance
        """
        Config.validate_config()
        self.config = Config
        self.logger = self._setup_logging()
        self.lightrag_instance = None
        self.rag_anything = None
        self.use_existing = use_existing_instance
        self._processed_files_cache = set()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_DIR / 'rag_manager.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize RAG system with existing or new LightRAG instance."""
        self.logger.info("Initializing RAG Manager...")
        
        try:
            # Check if existing LightRAG instance exists
            if (self.use_existing and 
                self.config.LIGHTRAG_STORAGE_DIR.exists() and 
                list(self.config.LIGHTRAG_STORAGE_DIR.glob("*"))):
                
                self.logger.info("Found existing LightRAG instance, loading...")
                await self._load_existing_lightrag()
            else:
                self.logger.info("Creating new LightRAG instance...")
                await self._create_new_lightrag()
            
            # Initialize RAGAnything with the LightRAG instance
            # Using README pattern: lightrag instance + vision function
            self.rag_anything = RAGAnything(
                lightrag=self.lightrag_instance,
                vision_model_func=self._get_vision_model_func()
            )
            
            # CRITICAL FIX: Manually configure modal processors with LLM functions
            self._fix_modal_processors()
            
            # Load processed files cache
            self._load_processed_files_cache()
            
            self.logger.info("RAG Manager initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Manager: {e}")
            raise
    
    async def _create_new_lightrag(self):
        """Create a new LightRAG instance."""
        self.lightrag_instance = LightRAG(
            working_dir=str(self.config.LIGHTRAG_STORAGE_DIR),
            llm_model_func=self._get_llm_model_func(),
            embedding_func=self._get_embedding_func()
        )
        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()  # 👈 CRITICAL: Initialize pipeline status
    
    async def _load_existing_lightrag(self):
        """Load existing LightRAG instance."""
        self.lightrag_instance = LightRAG(
            working_dir=str(self.config.LIGHTRAG_STORAGE_DIR),
            llm_model_func=self._get_llm_model_func(),
            embedding_func=self._get_embedding_func()
        )
        await self.lightrag_instance.initialize_storages()
        await initialize_pipeline_status()  # 👈 CRITICAL: Initialize pipeline status
    
    def _get_llm_model_func(self):
        """Get LLM model function."""
        def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                self.config.LLM_MODEL,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_BASE_URL,
                **kwargs
            )
        return llm_func
    
    def _get_vision_model_func(self):
        """Get vision model function."""
        def vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            if image_data:
                return openai_complete_if_cache(
                    self.config.VISION_MODEL,
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
                    api_key=self.config.OPENAI_API_KEY,
                    base_url=self.config.OPENAI_BASE_URL,
                    **kwargs
                )
            else:
                return openai_complete_if_cache(
                    self.config.LLM_MODEL,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.config.OPENAI_API_KEY,
                    base_url=self.config.OPENAI_BASE_URL,
                    **kwargs
                )
        return vision_func
    
    def _get_embedding_func(self):
        """Get embedding function."""
        return EmbeddingFunc(
            embedding_dim=self.config.EMBEDDING_DIM,
            max_token_size=self.config.MAX_TOKEN_SIZE,
            func=lambda texts: openai_embed(
                texts,
                model=self.config.EMBEDDING_MODEL,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_BASE_URL
            )
        )
    
    def _load_processed_files_cache(self):
        """Load cache of processed files."""
        cache_file = self.config.CACHE_DIR / "processed_files.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._processed_files_cache = set(json.load(f))
            except Exception as e:
                self.logger.warning(f"Could not load processed files cache: {e}")
                self._processed_files_cache = set()
        else:
            self._processed_files_cache = set()
    
    def _save_processed_files_cache(self):
        """Save cache of processed files."""
        cache_file = self.config.CACHE_DIR / "processed_files.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(list(self._processed_files_cache), f)
        except Exception as e:
            self.logger.warning(f"Could not save processed files cache: {e}")
    
    def _fix_modal_processors(self):
        """Fix modal processors with maximum data retention and robust JSON parsing."""
        try:
            if hasattr(self.rag_anything, 'modal_processors'):
                # Create a robust LLM function that preserves maximum data
                def robust_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                    try:
                        response = self._get_llm_model_func()(prompt, system_prompt, history_messages, **kwargs)
                        
                        # Ensure response exists
                        if response is None:
                            self.logger.warning("LLM returned None, using fallback response")
                            return '{"description": "Content processed but no analysis available", "entities": [], "content": "Unable to process content"}'
                        
                        if not isinstance(response, str):
                            response = str(response)
                        
                        response = response.strip()
                        original_response = response  # Keep original for data extraction
                        
                        # Step 1: Extract JSON from markdown blocks
                        if '```json' in response or '```' in response:
                            import re
                            json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                            if json_blocks:
                                response = json_blocks[0].strip()
                        
                        # Step 2: Find JSON object(s) in the response
                        import re
                        import json
                        
                        # Try to find complete JSON objects
                        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
                        json_matches = re.findall(json_pattern, response, re.DOTALL)
                        
                        valid_json = None
                        extra_content = ""
                        
                        # Test each JSON match for validity
                        for json_candidate in json_matches:
                            try:
                                parsed = json.loads(json_candidate)
                                valid_json = parsed
                                
                                # Extract any extra content that's not part of JSON
                                json_end = response.find(json_candidate) + len(json_candidate)
                                if json_end < len(response):
                                    extra_content = response[json_end:].strip()
                                break
                            except json.JSONDecodeError:
                                continue
                        
                        # Step 3: If no valid JSON found, try to construct one from content
                        if valid_json is None:
                            self.logger.info(f"No valid JSON found, constructing from content: {original_response[:100]}...")
                            
                            # Extract meaningful content for preservation
                            content_summary = original_response[:500]  # Preserve first 500 chars
                            
                            # Try to extract entities mentioned in the text
                            entities = []
                            # Look for common entity patterns
                            entity_patterns = [
                                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
                                r'\$[\d,]+\.?\d*',  # Money amounts
                                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
                                r'\b[A-Z]{2,}\b'  # Acronyms
                            ]
                            
                            for pattern in entity_patterns:
                                matches = re.findall(pattern, original_response)
                                entities.extend(matches[:5])  # Limit to prevent overflow
                            
                            # Construct valid JSON with preserved data
                            valid_json = {
                                "description": content_summary,
                                "entities": list(set(entities))[:10],  # Remove duplicates, limit size
                                "analysis": "Content processed with partial extraction",
                                "original_length": len(original_response)
                            }
                        
                        # Step 4: Enhance JSON with extra content if available
                        if extra_content and isinstance(valid_json, dict):
                            if "additional_notes" not in valid_json:
                                valid_json["additional_notes"] = extra_content[:200]  # Preserve extra content
                        
                        # Step 5: Ensure required fields exist
                        if isinstance(valid_json, dict):
                            if "description" not in valid_json:
                                valid_json["description"] = original_response[:200]
                            if "entities" not in valid_json:
                                valid_json["entities"] = []
                        
                        # Return clean JSON string
                        final_response = json.dumps(valid_json, ensure_ascii=False)
                        
                        self.logger.info(f"Successfully processed response with {len(valid_json.get('entities', []))} entities")
                        return final_response
                        
                    except Exception as e:
                        self.logger.error(f"Error in robust LLM function: {e}")
                        # Fallback with original content preservation
                        return json.dumps({
                            "description": f"Processing error occurred: {str(e)[:100]}",
                            "entities": [],
                            "error": True,
                            "original_content": str(response)[:300] if 'response' in locals() else "No content available"
                        })
                
                # Apply the robust function to all modal processors
                for processor_name, processor in self.rag_anything.modal_processors.items():
                    if hasattr(processor, 'modal_caption_func'):
                        if processor.modal_caption_func is None:
                            processor.modal_caption_func = robust_llm_func
                            self.logger.info(f"Enhanced {processor_name} processor with maximum data retention")
                        else:
                            # Wrap existing function to add robustness
                            original_func = processor.modal_caption_func
                            
                            def wrapped_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                                try:
                                    return robust_llm_func(prompt, system_prompt, history_messages, **kwargs)
                                except Exception:
                                    # Fallback to original if our enhancement fails
                                    return original_func(prompt, system_prompt, history_messages, **kwargs)
                            
                            processor.modal_caption_func = wrapped_func
                            self.logger.info(f"Wrapped {processor_name} processor with robust error handling")
                
                self.logger.info("All modal processors enhanced with maximum data retention")
            else:
                self.logger.warning("No modal_processors attribute found in RAGAnything")
                
        except Exception as e:
            self.logger.warning(f"Could not enhance modal processors: {e}")
            # Continue anyway - document processing can still work# Fix each modal processor
            for processor_name, processor in self.rag_anything.modal_processors.items():
                if hasattr(processor, 'modal_caption_func'):
                    if processor.modal_caption_func is None:
                        processor.modal_caption_func = robust_llm_func
                        self.logger.info(f"Fixed {processor_name} processor with robust LLM function")
                    else:
                        self.logger.info(f"{processor_name} processor already has LLM function")
                
                self.logger.info("Modal processors fixed successfully")
            else:
                self.logger.warning("No modal_processors attribute found in RAGAnything")
                
        except Exception as e:
            self.logger.warning(f"Could not fix modal processors: {e}")
            # Continue anyway - the document can still be processed without perfect modal processing
    
    async def process_document(self, file_path: str | Path, force_reprocess: bool = False) -> bool:
        """
        Process a single document.
        
        Args:
            file_path: Path to document
            force_reprocess: Force reprocessing even if already processed
            
        Returns:
            True if processed, False if skipped
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
        
        # Check if already processed
        if not force_reprocess and str(file_path) in self._processed_files_cache:
            self.logger.info(f"File already processed, skipping: {file_path}")
            return False
        
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            await self.rag_anything.process_document_complete(
                file_path=str(file_path),
                output_dir=str(self.config.PROCESSED_DOCS_DIR),
                parse_method=self.config.PARSE_METHOD
            )
            
            # Add to processed cache
            self._processed_files_cache.add(str(file_path))
            self._save_processed_files_cache()
            
            # Move file to processed directory if it's in pending
            if file_path.parent == self.config.PENDING_DIR:
                self.config.mark_file_as_processed(file_path)
            
            self.logger.info(f"Successfully processed: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {e}")
            return False
    
    async def process_pending_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all documents in the pending directory.
        
        Args:
            force_reprocess: Force reprocessing even if already processed
            
        Returns:
            Processing results summary
        """
        pending_files = list(self.config.PENDING_DIR.glob("*"))
        if not pending_files:
            self.logger.info("No pending documents found")
            return {"processed": 0, "skipped": 0, "failed": 0}
        
        results = {"processed": 0, "skipped": 0, "failed": 0}
        
        for file_path in pending_files:
            if file_path.is_file():
                success = await self.process_document(file_path, force_reprocess)
                if success:
                    results["processed"] += 1
                else:
                    if str(file_path) in self._processed_files_cache:
                        results["skipped"] += 1
                    else:
                        results["failed"] += 1
        
        self.logger.info(f"Processing complete: {results}")
        return results
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        Query the RAG system.
        
        Args:
            question: User question
            mode: Query mode (hybrid, local, global)
            
        Returns:
            Query result
        """
        try:
            self.logger.info(f"Querying: {question[:100]}...")
            
            result = await self.rag_anything.query_with_multimodal(
                question,
                mode=mode
            )
            
            self.logger.info("Query completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "lightrag_storage_exists": self.config.LIGHTRAG_STORAGE_DIR.exists(),
            "processed_files_count": len(self._processed_files_cache),
            "pending_files_count": len(list(self.config.PENDING_DIR.glob("*"))),
            "storage_directories": {
                "lightrag": str(self.config.LIGHTRAG_STORAGE_DIR),
                "processed_docs": str(self.config.PROCESSED_DOCS_DIR),
                "cache": str(self.config.CACHE_DIR)
            }
        }