"""
Documentation MCP Server
Handles internal documentation, wikis, and PDFs with production-grade search and RBAC.
"""

import asyncio
import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from core.base_server import BaseMCPServer, SearchResponse, SearchResult, MetadataResponse
from core.models import DataSourceType, UserRole
from core.permissions import PermissionManager
from config.embedding_config import get_embeddings, get_chunker

logger = logging.getLogger(__name__)

class DocumentationMCPServer(BaseMCPServer):
    """Production-grade MCP Server for internal documentation"""
    
    def __init__(self, name: str = "documentation", port: int = 8001, data_dir: str = "data/documentation"):
        super().__init__(name, port)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Initialize components
        self.embeddings = get_embeddings()
        # Pre-warm the embedding model to avoid first-request timeout
        self.logger.info("Warming up embedding model...")
        self.embeddings.initialize_model()
        
        self.chunker = get_chunker()
        self.permission_manager = PermissionManager()
        
        # Collection and Identity
        self.collection_name = "mcp_documentation"
        self.data_type = DataSourceType.DOCUMENTATION
        
        # Initialize collection
        self._ensure_collection_exists()
        
        # Document storage
        self.documents = {}
        self._load_source_documents()

    def _ensure_collection_exists(self):
        """Ensures the Qdrant collection is properly initialized"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embeddings.config.vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")

    def _load_source_documents(self):
        """Load documents from JSON source"""
        source_path = Path("data/store/documentation.json")
        if not source_path.exists():
            # Fallback for demo/dev
            source_path = self.data_dir / "documents.json"
            
        if source_path.exists():
            with open(source_path, 'r') as f:
                records = json.load(f)
                self.documents = {doc.get("id"): doc for doc in records}
                self.logger.info(f"Loaded {len(self.documents)} source documents")
        else:
            self.logger.warning("No documentation source records found.")

    async def search_documentation(
        self, 
        query: str, 
        user_role: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        MCP Tool: Search documentation with RBAC enforcement
        
        Args:
            query: The search query
            user_role: The role of the user (admin, engineer, product, support)
            max_results: Maximum results to return
        """
        # RBAC Check
        # Robust enum casting
        if isinstance(user_role, str):
            role = UserRole(user_role)
        elif isinstance(user_role, UserRole):
            role = user_role
        else:
            role = UserRole.ENGINEER
            
        if not self.permission_manager.can_access_source(role, self.data_type):
            self.logger.warning(f"Unauthorized access attempt by role: {user_role}")
            return {"error": "Unauthorized access to documentation", "results": []}

        try:
            # Generate embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Execute search via query_points (Modern Qdrant API)
            search_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=max_results,
                with_payload=True
            ).points
            
            # Format results
            formatted_results = []
            for hit in search_response:
                formatted_results.append({
                    "title": hit.payload.get("title", "Untitled"),
                    "content": hit.payload.get("content", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["content"]}
                })
            
            return {
                "results": formatted_results,
                "total_found": len(formatted_results),
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Search failure: {e}")
            return {"error": str(e), "results": []}

    async def get_documentation_freshness(self) -> Dict[str, Any]:
        """
        MCP Tool: Retrieve metadata and freshness info for documentation
        """
        if not self.documents:
            return {"error": "No documentation records available", "freshness_score": 0.0}
            
        now = datetime.now()
        total_days_since_update = 0
        latest_update = None
        
        for doc in self.documents.values():
            update_str = doc.get("last_updated", "2024-01-01T00:00:00Z")
            try:
                dt = datetime.fromisoformat(update_str.replace('Z', '+00:00')).replace(tzinfo=None)
                age = (now - dt).days
                total_days_since_update += age
                if latest_update is None or dt > latest_update:
                    latest_update = dt
            except Exception:
                continue
        
        avg_age = total_days_since_update / len(self.documents)
        # Freshness: 1.0 (recent) to 0.0 (old) based on 90 day window
        freshness_score = max(0.0, 1.0 - (avg_age / 90.0))
        
        return {
            "data_type": self.data_type.value,
            "document_count": len(self.documents),
            "freshness_score": round(freshness_score, 2),
            "average_age_days": round(avg_age, 1),
            "last_updated_record": latest_update.isoformat() if latest_update else None,
            "server_status": "healthy"
        }

    # Internal overrides for BaseMCPServer compatibility if needed
    async def search_documents(self, query: str, **kwargs) -> SearchResponse:
        """Compatibility override for BaseMCPServer"""
        role_input = kwargs.get("user_role", UserRole.ENGINEER)
        # Ensure it's a UserRole enum for the check
        if isinstance(role_input, str):
            role = UserRole(role_input)
        else:
            role = role_input
            
        limit = kwargs.get("max_results", 10)
        
        raw_result = await self.search_documentation(query, role.value, limit)
        if "error" in raw_result:
            return SearchResponse(results=[], total_found=0, search_time=0.0)
            
        results = [
            SearchResult(
                content=r["content"],
                source_id=r["metadata"].get("source_id", "unknown"),
                confidence_score=r["score"],
                metadata=r["metadata"]
            ) for r in raw_result["results"]
        ]
        return SearchResponse(results=results, total_found=len(results), search_time=0.0)

    async def get_metadata(self) -> MetadataResponse:
        """Compatibility override for BaseMCPServer"""
        freshness = await self.get_documentation_freshness()
        return MetadataResponse(
            document_count=freshness["document_count"],
            freshness_score=freshness["freshness_score"],
            last_updated=datetime.fromisoformat(freshness["last_updated_record"]) if freshness["last_updated_record"] else datetime.now(),
            data_type=self.data_type.value,
            ownership_info={
                "primary_owner": "documentation-team",
                "contact": "docs@company.com"
            }
        )

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        return self.documents.get(document_id)
