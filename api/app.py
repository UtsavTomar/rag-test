from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Any, Union
import logging

from knowledge.storage.knowledge_storage import KnowledgeStorage
from knowledge.knowledge import Knowledge
from knowledge.source.base_knowledge_source_search import BaseKnowledgeSourceSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Response model
class SearchResponse(BaseModel):
    query: str
    database_id: str
    results: Union[List[Any], str, dict]  # Allow list, string, or dict
    results_count: int
    status: str

def search_datasets_by_similarity(query_text: str, database_id: str) -> Union[List[Any], str, dict]:
    """Search for the most relevant datasets using vector similarity based on user query"""
    try:
        knowledge_storage = KnowledgeStorage(database_id=database_id)
        base_knowledge_source = BaseKnowledgeSourceSearch(storage=knowledge_storage)
        knowledge_test = Knowledge(
            storage=knowledge_storage,
            sources=[base_knowledge_source],
            database_id=database_id
        )
        result = knowledge_test.query([query_text])
        return result
    except Exception as e:
        logger.error(f"Error in search_datasets_by_similarity: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dataset Search API",
        "version": "1.0.0",
        "endpoints": {
            "/search": "Search datasets by similarity",
            "/health": "Health check endpoint"
        }
    }

@app.get("/search", response_model=SearchResponse)
async def search_datasets(
    query: str = Query(
        ..., 
        description="Search query for finding relevant datasets",
        min_length=1,
        max_length=500
    ),
    id: str = Query(
        "json_test",
        description="Database ID to search within",
        min_length=1,
        max_length=100,
        alias="id"
    )
):
    """
    Search for datasets using vector similarity based on the provided query.
    
    Args:
        query: The search query string
        id: The collection name/ID to search within (defaults to 'json_test')
        
    Returns:
        SearchResponse: Contains the query, collection_name, results, and status
    """
    try:
        logger.info(f"Received search request for query: '{query}' in collection: '{id}'")
        
        # Validate query
        if not query or query.strip() == "":
            raise HTTPException(
                status_code=400, 
                detail="Query parameter cannot be empty"
            )
        
        # Validate collection name
        if not id or id.strip() == "":
            raise HTTPException(
                status_code=400, 
                detail="Collection ID cannot be empty"
            )
        
        # Perform the search
        results = search_datasets_by_similarity(query.strip(), id.strip())
        
        # Calculate results count
        results_count = 0
        if isinstance(results, list):
            results_count = len(results)
        elif isinstance(results, dict):
            results_count = len(results) if results else 0
        elif isinstance(results, str) and results:
            results_count = 1
        
        logger.info(f"Search completed successfully for query: '{query}' in Database: '{id}' - Found {results_count} results")
        
        return SearchResponse(
            query=query,
            collection_name=id,
            results=results,
            results_count=results_count,
            status="success"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during search: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # You can add more sophisticated health checks here
        # For example, checking database connectivity
        return {
            "status": "healthy",
            "service": "Dataset Search API",
            "timestamp": "2024-01-01T00:00:00Z"  # You might want to use actual timestamp
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )



