from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Any, Union
import logging
import os
from datetime import datetime

# Import your knowledge modules - make sure these work in serverless environment
try:
    from knowledge.storage.knowledge_storage import KnowledgeStorage
    from knowledge.knowledge import Knowledge
    from knowledge.source.base_knowledge_source_search import BaseKnowledgeSourceSearch
    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Knowledge modules not available: {e}")
    KNOWLEDGE_AVAILABLE = False

# Configure logging for serverless
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dataset Search API",
    description="API for searching datasets using vector similarity",
    version="1.0.0",
    docs_url="/docs" if os.getenv("VERCEL_ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("VERCEL_ENV") != "production" else None
)

# Response models
class SearchResponse(BaseModel):
    query: str
    collection_name: str
    results: Union[List[Any], str, dict]
    results_count: int
    status: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    environment: str

# Global variables for caching
_knowledge_cache = {}

def get_knowledge_instance(collection_name: str):
    """Get or create knowledge instance with caching for performance"""
    if not KNOWLEDGE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Knowledge modules not available"
        )
    
    cache_key = f"knowledge_{collection_name}"
    if cache_key not in _knowledge_cache:
        try:
            knowledge_storage = KnowledgeStorage(collection_name=collection_name)
            base_knowledge_source = BaseKnowledgeSourceSearch(storage=knowledge_storage)
            knowledge_instance = Knowledge(
                storage=knowledge_storage,
                sources=[base_knowledge_source],
                collection_name=collection_name
            )
            _knowledge_cache[cache_key] = knowledge_instance
            logger.info(f"Created new knowledge instance for collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create knowledge instance: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize knowledge storage: {str(e)}"
            )
    
    return _knowledge_cache[cache_key]

def search_datasets_by_similarity(query_text: str, collection_name: str) -> Union[List[Any], str, dict]:
    """Search for the most relevant datasets using vector similarity"""
    try:
        knowledge_instance = get_knowledge_instance(collection_name)
        result = knowledge_instance.query([query_text])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_datasets_by_similarity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dataset Search API",
        "version": "1.0.0",
        "status": "running",
        "environment": os.getenv("VERCEL_ENV", "development"),
        "knowledge_available": KNOWLEDGE_AVAILABLE,
        "endpoints": {
            "/api/search": "Search datasets by similarity",
            "/api/health": "Health check endpoint",
        }
    }

@app.get("/api/search", response_model=SearchResponse)
async def search_datasets(
    request: Request,
    query: str = Query(
        ..., 
        description="Search query for finding relevant datasets",
        min_length=1,
        max_length=500
    ),
    id: str = Query(
        "json_test",
        description="Collection name/ID to search within",
        min_length=1,
        max_length=100,
        alias="id"
    )
):
    """Search for datasets using vector similarity"""
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Search request - Query: '{query}' Collection: '{id}'")
        
        if not query or query.strip() == "":
            raise HTTPException(status_code=400, detail="Query parameter cannot be empty")
        
        if not id or id.strip() == "":
            raise HTTPException(status_code=400, detail="Collection ID cannot be empty")
        
        if not KNOWLEDGE_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge search service is temporarily unavailable"
            )
        
        results = search_datasets_by_similarity(query.strip(), id.strip())
        
        results_count = 0
        if isinstance(results, list):
            results_count = len(results)
        elif isinstance(results, dict):
            results_count = len(results) if results else 0
        elif isinstance(results, str) and results:
            results_count = 1
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Search completed - Duration: {duration:.2f}s Results: {results_count}")
        
        return SearchResponse(
            query=query,
            collection_name=id,
            results=results,
            results_count=results_count,
            status="success",
            timestamp=end_time.isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during search"
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            service="Dataset Search API",
            timestamp=datetime.utcnow().isoformat() + "Z",
            environment=os.getenv("VERCEL_ENV", "development")
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found", 
            "detail": "The requested endpoint does not exist",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": "An unexpected error occurred"
        }
    )

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# This is the handler that Vercel will use
def handler(request):
    """Vercel serverless handler"""
    from mangum import Mangum
    asgi_handler = Mangum(app)
    return asgi_handler(request, {})