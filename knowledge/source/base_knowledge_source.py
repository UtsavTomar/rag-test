from abc import ABC, abstractmethod 
from typing import Any, Dict, List, Optional 
import numpy as np 
from pydantic import BaseModel, ConfigDict, Field 
from knowledge.storage.knowledge_storage import KnowledgeStorage 

class BaseKnowledgeSource(BaseModel, ABC): 
    """Abstract base class for knowledge sources.""" 
    chunk_size: int = 4000 
    chunk_overlap: int = 200 
    chunks: List[str] = Field(default_factory=list) 
    chunk_embeddings: List[np.ndarray] = Field(default_factory=list) 
    model_config = ConfigDict(arbitrary_types_allowed=True) 
    storage: Optional[KnowledgeStorage] = Field(default=None) 
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Currently unused 
    collection_name: Optional[str] = Field(default=None)
    # New field for existing table name
    table_name: Optional[str] = Field(default=None)
    use_existing_chunks: bool = Field(default=False)
     
    @abstractmethod
    def validate_content(self) -> Any: 
        """Load and preprocess content from the source.""" 
        pass 
   
    @abstractmethod
    def add(self) -> None: 
        """Process content, chunk it, compute embeddings, and save them.""" 
        pass
    
    def load_existing_chunks(self) -> None:
        """Load chunks and embeddings from existing PG vector table."""
        if not self.table_name:
            raise ValueError("table_name must be provided to load existing chunks")
        if not self.storage:
            raise ValueError("Storage must be configured to load existing chunks")
        # Load chunks and embeddings from the specified table
        chunks, embeddings = self.storage.load_from_table(self.table_name)
        self.chunks = chunks
        self.chunk_embeddings = embeddings
    def initialize_from_table(self, table_name: str) -> None:
        """Initialize the knowledge source from an existing PG vector table."""
        self.table_name = table_name
        self.use_existing_chunks = True
        self.load_existing_chunks()
    def get_embeddings(self) -> List[np.ndarray]: 
        """Return the list of embeddings for the chunks.""" 
        if self.use_existing_chunks and not self.chunk_embeddings:
            self.load_existing_chunks()
        return self.chunk_embeddings 
    def _chunk_text(self, text: str) -> List[str]: 
        """Utility method to split text into chunks.""" 
        return [ 
            text[i : i + self.chunk_size] 
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap) 
        ] 
    def _save_documents(self): 
        """ 
        Save the documents to the storage. 
        This method should be called after the chunks and embeddings are generated. 
        """ 
        if self.storage: 
            if self.use_existing_chunks:
                # If using existing chunks, we don't need to save again
                # unless you want to update/sync the storage
                print(f"Using existing chunks from table: {self.table_name}")
            else:
                self.storage.save(self.chunks) 
        else: 
            raise ValueError("No storage found to save documents.")