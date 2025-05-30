import json
from pathlib import Path
from typing import Any, Dict, List
from pydantic import Field
from knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource

class JSONKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries JSON file content using embeddings."""
    
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    chunks: List[str] = Field(default_factory=list, description="Text chunks from JSON content")
    
    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess JSON file content."""
        content = {}
        for file_path in self.safe_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    # Convert JSON to readable text format
                    text_content = self._json_to_text(json_data)
                    content[file_path] = text_content
                    self._logger.log("info", f"Successfully loaded JSON from {file_path}")
            except json.JSONDecodeError as e:
                self._logger.log("error", f"Invalid JSON in file {file_path}: {e}", color="red")
                raise
            except Exception as e:
                self._logger.log("error", f"Error loading file {file_path}: {e}", color="red")
                raise
        return content
    
    def _json_to_text(self, data: Any, level: int = 0) -> str:
        """Recursively convert JSON data to a text representation."""
        text = ""
        indent = "  " * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text += f"{indent}{key}:\n{self._json_to_text(value, level + 1)}\n"
                else:
                    text += f"{indent}{key}: {self._json_to_text(value, level + 1)}\n"
        elif isinstance(data, list):
            for i, item in enumerate(data):
                text += f"{indent}[{i}] {self._json_to_text(item, level + 1)}\n"
        else:
            return str(data)
        
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap consideration
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks
    
    def add(self) -> None:
        """
        Add JSON file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        # Clear existing chunks
        self.chunks = []
        
        # Process all loaded content
        for file_path, content_text in self.content.items():
            self._logger.log("info", f"Processing {file_path}")
            
            # Chunk the text content
            file_chunks = self._chunk_text(content_text)
            
            # Add metadata to chunks if needed (optional)
            enhanced_chunks = [
                f"Source: {file_path.name}\n\n{chunk}" 
                for chunk in file_chunks
            ]
            
            self.chunks.extend(enhanced_chunks)
        
        # Save to storage
        self._save_documents()
        self._logger.log("info", f"Added {len(self.chunks)} chunks to storage")
    
    def update(self) -> None:
        """
        Update the knowledge source by reloading content and updating storage.
        """
        self._logger.log("info", "Updating JSON knowledge source...")
        
        # Reload content from files
        self.content = self.load_content()
        
        # Re-add content (this will clear and rebuild chunks)
        self.add()
        
        self._logger.log("info", "Successfully updated JSON knowledge source")
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded content."""
        summary = {
            "total_files": len(self.content),
            "total_chunks": len(self.chunks),
            "files": []
        }
        
        for file_path, content in self.content.items():
            summary["files"].append({
                "path": str(file_path),
                "content_length": len(content),
                "chunks_count": len(self._chunk_text(content))
            })
        
        return summary