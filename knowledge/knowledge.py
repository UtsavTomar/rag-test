import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from knowledge.source.base_knowledge_source import BaseKnowledgeSource
from knowledge.storage.knowledge_storage import KnowledgeStorage

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    Args:
        sources: List[BaseKnowledgeSource] = Field(default_factory=list)
        storage: Optional[KnowledgeStorage] = Field(default=None)
        embedder: Optional[Dict[str, Any]] = None
    """

    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    embedder: Optional[Dict[str, Any]] = None
    dataset_metadata_id: Optional[str] = None

    def __init__(
        self,
        dataset_metadata_id: str,
        sources: List[BaseKnowledgeSource],
        embedder: Optional[Dict[str, Any]] = None,
        storage: Optional[KnowledgeStorage] = None,
        **data,
    ):
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder=embedder, dataset_metadata_id=dataset_metadata_id
            )
        self.sources = sources
        self.storage.initialize_knowledge_storage()

    def query(
        self, query: List[str], results_limit: int = 3, score_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.

        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage is not initialized.")

        results = self.storage.search(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )
        return results

    def add_sources(self):
        try:
            for source in self.sources:
                source.storage = self.storage
                source.add()
        except Exception as e:
            raise e

    def reset(self) -> None:
        if self.storage:
            self.storage.reset()
        else:
            raise ValueError("Storage is not initialized.")
