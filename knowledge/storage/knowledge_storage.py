import contextlib
import hashlib
import io
import logging
import os
import json
from typing import Any, Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
 
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
 
 
class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries using PostgreSQL with pgvector,
    improving search efficiency.
    """
 
    connection: Optional[psycopg2.extensions.connection] = None
    collection_name: str = "knowledge"
    table_name: str = "knowledge_embeddings"
 
    def __init__(
        self,
        embedder: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        db_host: str = "ep-broad-thunder-a5ebmxk0-pooler.us-east-2.aws.neon.tech",
        db_port: int = 5432,
        db_name: str = "neondb",
        db_user: str = "neondb_owner",
        db_password: str = "npg_hLdNTex4KUi9",
    ):
        self.collection_name = collection_name or "knowledge"
        self.table_name = self._sanitize_table_name(f"{self.collection_name}_embeddings")
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
        }
        self._set_embedder_config(embedder)
 
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize table name to be PostgreSQL compliant."""
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"tbl_{sanitized}"
        # Ensure it's not empty
        if not sanitized:
            sanitized = "knowledge_embeddings"
        # Limit length to 63 characters (PostgreSQL limit)
        return sanitized[:63].lower()
 
    def _get_connection(self):
        """Get database connection, creating one if it doesn't exist."""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.db_config)
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
           
            # Build the SQL query with properly quoted table name
            sql = f"""
                SELECT id, metadata, context,
                       1 - (embedding <=> %s::vector) as score
                FROM "{self.table_name}"
                WHERE 1 = 1
            """
            params = [query_embedding]
           
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
               
                # Create the embeddings table with properly quoted table name
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                        id TEXT PRIMARY KEY,
                        context TEXT NOT NULL,
                        metadata JSONB,
                        embedding VECTOR(1536),  -- Adjust dimension based on your embedding model
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
               
                # Create index for vector similarity search
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS "{self.table_name}_embedding_idx"
                    ON "{self.table_name}" USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
               
                # Create index on metadata for filtering
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS "{self.table_name}_metadata_idx"
                    ON "{self.table_name}" USING gin (metadata);
                """)
               
                conn.commit()
                Logger(verbose=True).log("info", f"Knowledge storage initialized with table: {self.table_name}", "green")
               
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to initialize knowledge storage: {e}", "red")
            raise Exception(f"Failed to initialize knowledge storage: {e}")
 
    def reset(self):
        """Reset the knowledge storage by dropping the table."""
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute(f'DROP TABLE IF EXISTS "{self.table_name}";')
                    self.connection.commit()
               
                self.connection.close()
                self.connection = None
               
                Logger(verbose=True).log("info", f"Knowledge storage reset: {self.table_name} dropped", "green")
               
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to reset knowledge storage: {e}", "red")
            raise
 
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
                doc_id = hashlib.sha256(doc.encode("utf-8")).hexdigest()
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
                   
                    # Upsert the document with properly quoted table name
                    cursor.execute(f"""
                        INSERT INTO "{self.table_name}" (id, context, metadata, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        ON CONFLICT (id)
                        DO UPDATE SET
                            context = EXCLUDED.context,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP;
                    """, (doc_id, doc, metadata_json, embedding))
               
                self.connection.commit()
                Logger(verbose=True).log("info", f"Saved {len(unique_docs)} unique documents", "green")
               
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
                        model=self.model
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
 