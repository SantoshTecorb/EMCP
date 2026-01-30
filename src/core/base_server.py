from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    min_confidence: float = 0.7

    class Config:
        extra = "allow"


class SearchResult(BaseModel):
    content: str
    source_id: str
    confidence_score: float
    metadata: Dict[str, Any] = {}
    vector_id: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_found: int
    search_time: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


class MetadataResponse(BaseModel):
    document_count: int
    freshness_score: float
    last_updated: datetime
    data_type: str
    ownership_info: Dict[str, Any] = Field(default_factory=dict)


class BaseMCPServer(ABC):
    """Base class for MCP context servers"""
    
    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self.app = FastAPI(title=f"{name} MCP Server")
        self.setup_routes()
        self.logger = logging.getLogger(name)
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check_route():
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="1.0.0"
            )
        
        @self.app.get("/metadata", response_model=MetadataResponse)
        async def get_metadata_route():
            return await self.get_metadata()
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search_route(request: SearchRequest):
            try:
                params = request.model_dump()
                query = params.pop("query")
                return await self.search_documents(query, **params)
            except Exception as e:
                self.logger.error(f"Search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/documents/{document_id}")
        async def get_document_route(document_id: str):
            try:
                doc = await self.get_document_by_id(document_id)
                if doc is None:
                    raise HTTPException(status_code=404, detail="Document not found")
                return doc
            except Exception as e:
                self.logger.error(f"Document retrieval error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    @abstractmethod
    async def search_documents(
        self, 
        query: str, 
        **kwargs
    ) -> SearchResponse:
        """Search documents in the context server"""
        pass
    
    @abstractmethod
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        pass
    
    @abstractmethod
    async def get_metadata(self) -> MetadataResponse:
        """Get server metadata"""
        pass
    
    def run(self, host: str = "0.0.0.0"):
        """Run the server"""
        uvicorn.run(self.app, host=host, port=self.port)
    
    async def run_async(self, host: str = "0.0.0.0"):
        """Run the server asynchronously"""
        config = uvicorn.Config(self.app, host=host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
