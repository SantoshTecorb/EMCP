"""
Runbooks MCP Server
Handles engineering procedures, deployment guides, and operational runbooks with system filtering and RBAC.
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
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)

from core.base_server import BaseMCPServer, SearchResponse, SearchResult, MetadataResponse
from core.models import DataSourceType, UserRole
from core.permissions import PermissionManager
from config.embedding_config import get_embeddings

logger = logging.getLogger(__name__)

class RunbooksMCPServer(BaseMCPServer):
    """Production-grade MCP Server for engineering runbooks and procedures"""
    
    def __init__(self, name: str = "runbooks", port: int = 8003, data_dir: str = "data/runbooks"):
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
        
        self.permission_manager = PermissionManager()
        
        # Collection and Identity
        self.collection_name = "mcp_runbooks"
        self.data_type = DataSourceType.RUNBOOKS
        
        # Initialize collection
        self._ensure_collection_exists()
        
        # Document storage
        self.runbooks = {}
        self._load_source_data()

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

    def _load_source_data(self):
        """Load data from JSON source"""
        source_path = Path("data/store/runbooks.json")
        if not source_path.exists():
            source_path = self.data_dir / "runbooks.json"
            
        if source_path.exists():
            with open(source_path, 'r') as f:
                records = json.load(f)
                self.runbooks = {rec.get("id"): rec for rec in records}
                self.logger.info(f"Loaded {len(self.runbooks)} runbook records")
        else:
            self.logger.warning("No runbook source records found.")

    async def search_runbooks(
        self, 
        query: str, 
        user_role: str,
        system: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        MCP Tool: Search engineering runbooks with system filtering and RBAC
        
        Args:
            query: The search query
            user_role: The role of the user (admin, engineer, devops, leadership)
            system: Filter by system or category (e.g., "database", "deployment", "security")
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
            return {"error": "Unauthorized access to runbooks", "results": []}

        try:
            # Construct Filter logic
            must_conditions = []
            if system:
                # Map system filter to either category or tags
                must_conditions.append(
                    Filter(
                        should=[
                            FieldCondition(key="category", match=MatchValue(value=system)),
                            FieldCondition(key="tags", match=MatchValue(value=system)),
                            FieldCondition(key="team", match=MatchValue(value=system))
                        ]
                    )
                )
            
            search_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Generate embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Execute search
            search_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=max_results,
                with_payload=True
            ).points
            
            # Format results
            formatted_results = []
            for hit in search_response:
                formatted_results.append({
                    "id": hit.payload.get("source_id"),
                    "title": hit.payload.get("title", "Untitled"),
                    "content": hit.payload.get("content", ""),
                    "category": hit.payload.get("category"),
                    "system": hit.payload.get("system"),
                    "owner": hit.payload.get("owner"),
                    "score": hit.score
                })
            
            return {
                "results": formatted_results,
                "total_found": len(formatted_results),
                "applied_system_filter": system,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Search failure: {e}")
            return {"error": str(e), "results": []}

    async def get_metadata(self) -> MetadataResponse:
        """Compatibility override for BaseMCPServer"""
        return MetadataResponse(
            document_count=len(self.runbooks),
            freshness_score=self._calculate_freshness(),
            last_updated=datetime.now(),
            data_type=self.data_type.value,
            ownership_info={
                "primary_owner": "engineering-team",
                "contact": "engineering@company.com"
            }
        )

    def _calculate_freshness(self) -> float:
        """Calculate freshness score"""
        if not self.runbooks: return 0.0
        now = datetime.now()
        total_age = 0
        for rec in self.runbooks.values():
            updated = datetime.fromisoformat(rec.get("last_updated", "2024-01-01T00:00:00Z").replace('Z', '+00:00')).replace(tzinfo=None)
            total_age += (now - updated).days
        avg_age = total_age / len(self.runbooks)
        return max(0.0, 1.0 - (avg_age / 90.0))

    async def search_documents(self, query: str, **kwargs) -> SearchResponse:
        """BaseMCPServer compatibility"""
        role_input = kwargs.get("user_role", UserRole.ENGINEER)
        if isinstance(role_input, str):
            role = UserRole(role_input)
        else:
            role = role_input
            
        limit = kwargs.get("max_results", 10)
        system = kwargs.get("system")
        
        raw = await self.search_runbooks(query, role.value, system=system, max_results=limit)
        if "error" in raw: return SearchResponse(results=[], total_found=0, search_time=0.0)
        
        results = [
            SearchResult(
                content=r["content"],
                source_id=r["id"] or "unknown",
                confidence_score=r["score"],
                metadata=r
            ) for r in raw["results"]
        ]
        return SearchResponse(results=results, total_found=len(results), search_time=0.0)

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific runbook by ID"""
        return self.runbooks.get(document_id)
