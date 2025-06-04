import contextlib
import hashlib
import io
import logging
import os
import json
import psycopg2
import numpy as np
import uuid
from typing import Any, Dict, List, Optional, Union
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from typing import Optional, Dict, Any
 
 
from knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from knowledge.utilities import EmbeddingConfigurator
from knowledge.utilities.logger import Logger
 
 
@contextlib.contextmanager
def suppress_logging(
    logger_name="psycopg2",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)
 
 
def get_db_connection():
    """
    Create and return a database connection.
   
    Returns:
        psycopg2.connection: A connection to the PostgreSQL database.
       
    Raises:
        Exception: If the database connection string is not found or connection fails.
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise Exception("Database url not found")
   
    try:
        connection = psycopg2.connect(DATABASE_URL)
        return connection
    except Exception as e:
        raise
 
class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries using PostgreSQL with pgvector,
    improving search efficiency.
    """
 
    connection: Optional[psycopg2.extensions.connection] = None
    table_name: str = '"agentic-platform"."DatasetVectorStoreRecords"'
 
 
    def __init__(
        self,
        dataset_metadata_id: str,
        embedder: Optional[Dict[str, Any]] = None,
        db_url: str = os.getenv("DATABASE_URL")
    ):
        self.dataset_metadata_id = dataset_metadata_id
        self.table_name = '"agentic-platform"."DatasetVectorStoreRecords"'
       
        # Define base table name for index creation (without quotes)
        self.base_table_name = "agentic_platform_DatasetVectorStoreRecords"
       
        # Store the full URL as well
        self.db_url = db_url
       
        self._set_embedder_config(embedder)
 
    def _get_connection(self):
        """Get database connection, creating one if it doesn't exist."""
        if self.connection is None or self.connection.closed:
            self.connection = get_db_connection()
        return self.connection
 
    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        with suppress_logging():
            if not self.connection:
                raise Exception("Database connection not initialized")
           
            # Get embedding for the query
            query_text = " ".join(query)
            query_embedding = self._get_embedding(query_text)
           
            # Build the SQL query with dataset_metadata_id filter
            sql = f"""
                SELECT id, metadata, context, dataset_metadata_id,
                       1 - (embedding <=> %s::vector) as score
                FROM {self.table_name}
                WHERE dataset_metadata_id = %s
            """
            params = [query_embedding, self.dataset_metadata_id]
           
            # Add metadata filter if provided
            if filter:
                for key, value in filter.items():
                    sql += f" AND metadata->>%s = %s"
                    params.extend([key, str(value)])
           
            # Add score threshold and limit
            sql += f" AND (1 - (embedding <=> %s::vector)) >= %s"
            sql += f" ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, score_threshold, query_embedding, limit])
           
            try:
                with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                   
                    results = []
                    for row in rows:
                        result = {
                            "id": row["id"],
                            "metadata": row["metadata"],
                            "context": row["context"],
                            "dataset_metadata_id": row["dataset_metadata_id"],
                            "score": float(row["score"]),
                        }
                        results.append(result)
                   
                    return results
            except Exception as e:
                Logger(verbose=True).log("error", f"Search failed: {e}", "red")
                raise
 
    def initialize_knowledge_storage(self):
        """Initialize the PostgreSQL database and create necessary tables."""
        try:
            conn = self._get_connection()
           
            with conn.cursor() as cursor:
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
               
                # Create the embeddings table with dataset_metadata_id column
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id uuid PRIMARY KEY,
                        dataset_metadata_id uuid NOT NULL,
                        context TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR(1536),  -- Adjust dimension based on your embedding model
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
               
                # Create index for vector similarity search with dataset_metadata_id
                # Using properly quoted index names
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS "{self.base_table_name}_embedding_dataset_metadata_idx"
                    ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                    WHERE dataset_metadata_id IS NOT NULL;
                """)
               
                # Create index on dataset_metadata_id for filtering
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS "{self.base_table_name}_dataset_metadata_id_idx"
                    ON {self.table_name} (dataset_metadata_id);
                """)
               
                # Create index on metadata for filtering
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS "{self.base_table_name}_metadata_idx"
                    ON {self.table_name} USING gin (metadata);
                """)
               
                conn.commit()
                Logger(verbose=True).log("info", f"Knowledge storage initialized with table: {self.table_name} and dataset_metadata_id: {self.dataset_metadata_id}", "green")
               
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to initialize knowledge storage: {e}", "red")
            raise Exception(f"Failed to initialize knowledge storage: {e}")
 
    def reset(self, reset_all: bool = False):
        """Reset the knowledge storage by dropping records for this collection or entire table."""
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    if reset_all:
                        # Drop entire table
                        cursor.execute(f'DROP TABLE IF EXISTS {self.table_name};')
                        Logger(verbose=True).log("info", f"Entire table {self.table_name} dropped", "green")
                    else:
                        # Delete only records for this dataset_metadata_id
                        cursor.execute(f'DELETE FROM {self.table_name} WHERE dataset_metadata_id = %s;', (self.dataset_metadata_id,))
                        Logger(verbose=True).log("info", f"Records for dataset_metadata_id {self.dataset_metadata_id} deleted", "green")
                   
                    self.connection.commit()
               
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to reset knowledge storage: {e}", "red")
            raise
 
    def _generate_uuid_from_content(self, content: str) -> str:
        """Generate a deterministic UUID from content using SHA-256 hash."""
        # Create a SHA-256 hash of the content
        hash_obj = hashlib.sha256(content.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
       
        # Convert the first 32 characters of the hash to UUID format
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
       
        return uuid_str
 
    def save(
        self,
        documents: List[str],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """Save documents with their embeddings to PostgreSQL."""
        if not self.connection:
            raise Exception("Database connection not initialized")
 
        try:
            # Create a dictionary to store unique documents
            unique_docs = {}
 
            # Generate IDs and create a mapping of id -> (document, metadata)
            for idx, doc in enumerate(documents):
                # Generate a UUID from the document content
                doc_id = self._generate_uuid_from_content(doc)
                doc_metadata = None
                if metadata is not None:
                    if isinstance(metadata, list):
                        doc_metadata = metadata[idx] if idx < len(metadata) else None
                    else:
                        doc_metadata = metadata
                unique_docs[doc_id] = (doc, doc_metadata)
 
            # Process each unique document
            with self.connection.cursor() as cursor:
                for doc_id, (doc, meta) in unique_docs.items():
                    # Get embedding for the document
                    embedding = self._get_embedding(doc)
                   
                    # Convert metadata to JSON
                    metadata_json = json.dumps(meta) if meta else None
                   
                    # Upsert the document with dataset_metadata_id
                    cursor.execute(f"""
                        INSERT INTO {self.table_name} (id, dataset_metadata_id, context, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id)
                        DO UPDATE SET
                            context = EXCLUDED.context,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP;
                    """, (doc_id, self.dataset_metadata_id, doc, metadata_json, embedding))
 
               
                self.connection.commit()
                Logger(verbose=True).log("info", f"Saved {len(unique_docs)} unique documents for collection {self.dataset_metadata_id}", "green")
               
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            Logger(verbose=True).log("error", f"Failed to save documents: {e}", "red")
            raise
 
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using the configured embedder."""
        try:
            if hasattr(self.embedder, '__call__'):
                # If embedder is callable (like OpenAI function)
                result = self.embedder([text])
                if isinstance(result, list) and len(result) > 0:
                    return result[0]
                return result
            else:
                # If embedder has an embed method
                return self.embedder.embed([text])[0]
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to get embedding: {e}", "red")
            raise
 
    def _create_default_embedding_function(self):
        """Create default OpenAI embedding function."""
        try:
            import openai
           
            class OpenAIEmbeddingFunction:
                def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model = model
               
                def __call__(self, texts: List[str]) -> List[List[float]]:
                    response = self.client.embeddings.create(
                        input=texts,
                        model=self.model,
                    )
                    return [data.embedding for data in response.data]
           
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
               
            return OpenAIEmbeddingFunction(api_key=api_key)
           
        except ImportError:
            raise Exception("OpenAI package is required for default embedding function")
 
    def _set_embedder_config(self, embedder: Optional[Dict[str, Any]] = None) -> None:
        """Set the embedding configuration for the knowledge storage.
 
        Args:
            embedder_config (Optional[Dict[str, Any]]): Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder = (
            EmbeddingConfigurator().configure_embedder(embedder)
            if embedder
            else self._create_default_embedding_function()
        )
 
    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'connection') and self.connection and not self.connection.closed:
            self.connection.close()
 
 