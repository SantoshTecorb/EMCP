"""
Tickets MCP Server
Handles support tickets, CRM data, and customer interactions with advanced filtering and RBAC.
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
    Filter, FieldCondition, MatchValue, Range
)

from core.base_server import BaseMCPServer, SearchResponse, SearchResult, MetadataResponse
from core.models import DataSourceType, UserRole
from core.permissions import PermissionManager
from config.embedding_config import get_embeddings
try:
    from fastapi import HTTPException
except ImportError:
    HTTPException = None

logger = logging.getLogger(__name__)

class TicketsMCPServer(BaseMCPServer):
    """Production-grade MCP Server for support tickets and CRM data"""
    
    def __init__(self, name: str = "tickets", port: int = 8002, data_dir: str = "data/tickets"):
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
        self.collection_name = "mcp_tickets"
        self.data_type = DataSourceType.TICKETS
        
        # Initialize collection
        self._ensure_collection_exists()
        
        # Internal storage
        self.tickets = {}
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
        source_path = Path("data/store/tickets.json")
        if not source_path.exists():
            source_path = self.data_dir / "tickets.json"
            
        if source_path.exists():
            with open(source_path, 'r') as f:
                records = json.load(f)
                self.tickets = {rec.get("id"): rec for rec in records}
                self.logger.info(f"Loaded {len(self.tickets)} ticket/CRM records")
        else:
            self.logger.warning("No ticket/CRM source records found.")

    def _extract_ticket_id(self, query: str) -> Optional[str]:
        """Extract ticket ID from query if present"""
        import re
        # Look for patterns like "ticket-022", "ticket 022", "ticket022"
        patterns = [
            r'ticket-(\d+)',
            r'ticket\s*(\d+)',
            r'ticket(\d+)',
            r'tk-(\d+)',
            r'tk\s*(\d+)',
            r'tk(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return f"ticket-{match.group(1).zfill(3)}"  # Format as ticket-XXX
        
        return None

    async def search_tickets(
        self, 
        query: str, 
        user_role: str,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        MCP Tool: Search tickets and CRM data with advanced filtering and RBAC
        
        Args:
            query: The search query
            user_role: The role of the user (admin, support, product, leadership)
            status: Filter by status (e.g., "open", "resolved", "pending")
            start_date: Filter by date range (ISO format start)
            end_date: Filter by date range (ISO format end)
            max_results: Maximum results to return
        """
        # RBAC Check
        # Robust enum casting
        if isinstance(user_role, str):
            role = UserRole(user_role)
        elif isinstance(user_role, UserRole):
            role = user_role
        else:
            role = UserRole.SUPPORT

        if not self.permission_manager.can_access_source(role, self.data_type):
            self.logger.warning(f"Unauthorized access attempt by role: {user_role}")
            return {"error": "Unauthorized access to ticket/CRM data", "results": []}
        
        # Check for direct ticket ID lookup first
        ticket_id = self._extract_ticket_id(query)
        self.logger.info(f"Extracted ticket_id: {ticket_id} from query: '{query}'")
        
        if ticket_id and ticket_id in self.tickets:
            self.logger.info(f"Direct ticket lookup found: {ticket_id}")
            ticket = self.tickets[ticket_id]
            self.logger.info(f"Returning ticket content: {ticket['title'][:50]}...")
            return {
                "results": [{
                    "id": ticket["id"],
                    "title": ticket["title"],
                    "content": ticket["content"],
                    "status": ticket["status"],
                    "customer": ticket["customer"],
                    "created_date": ticket["created_date"],
                    "score": 1.0,  # Perfect match for direct ID lookup
                    "lookup_method": "direct_id"
                }],
                "total_found": 1,
                "applied_filters": {
                    "ticket_id": ticket_id,
                    "lookup_method": "direct"
                }
            }

        try:
            # Construct Filter logic
            must_conditions = []
            
            if status:
                must_conditions.append(
                    FieldCondition(key="status", match=MatchValue(value=status))
                )
                
            if start_date or end_date:
                start_dt = datetime.fromisoformat(start_date) if start_date else None
                end_dt = datetime.fromisoformat(end_date) if end_date else None
                must_conditions.append(
                    FieldCondition(
                        key="created_date", 
                        range=Range(
                            gte=start_dt,
                            lte=end_dt
                        )
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
                    "status": hit.payload.get("status"),
                    "customer": hit.payload.get("customer"),
                    "created_date": hit.payload.get("created_date"),
                    "score": hit.score
                })
            
            # Fallback: If vector search results are poor and we have a ticket ID, try direct lookup
            avg_score = sum(r["score"] for r in formatted_results) / len(formatted_results) if formatted_results else 0
            if avg_score < 0.3 and ticket_id and ticket_id in self.tickets:
                self.logger.info(f"Vector search poor (avg score: {avg_score:.2f}), falling back to direct lookup for {ticket_id}")
                ticket = self.tickets[ticket_id]
                fallback_result = {
                    "id": ticket["id"],
                    "title": ticket["title"],
                    "content": ticket["content"],
                    "status": ticket["status"],
                    "customer": ticket["customer"],
                    "created_date": ticket["created_date"],
                    "score": 1.0,
                    "lookup_method": "direct_fallback"
                }
                return {
                    "results": [fallback_result],
                    "total_found": 1,
                    "applied_filters": {
                        "status": status,
                        "date_range": f"{start_date} to {end_date}" if start_date or end_date else "None",
                        "fallback_used": "direct_id_lookup"
                    }
                }
            
            return {
                "results": formatted_results,
                "total_found": len(formatted_results),
                "applied_filters": {
                    "status": status,
                    "date_range": f"{start_date} to {end_date}" if start_date or end_date else "None"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search failure: {e}")
            return {"error": str(e), "results": []}

    async def get_metadata(self) -> MetadataResponse:
        """Compatibility override for BaseMCPServer"""
        return MetadataResponse(
            document_count=len(self.tickets),
            freshness_score=self._calculate_freshness(),
            last_updated=datetime.now(),
            data_type=self.data_type.value,
            ownership_info={
                "primary_owner": "support-team",
                "contact": "support@company.com"
            }
        )

    def _calculate_freshness(self) -> float:
        """Calculate freshness score"""
        if not self.tickets: return 0.0
        now = datetime.now()
        total_age = 0
        for rec in self.tickets.values():
            updated = datetime.fromisoformat(rec.get("last_updated", "2024-01-01T00:00:00Z").replace('Z', '+00:00')).replace(tzinfo=None)
            total_age += (now - updated).days
        avg_age = total_age / len(self.tickets)
        return max(0.0, 1.0 - (avg_age / 30.0))

    async def search_documents(self, query: str, **kwargs) -> SearchResponse:
        """BaseMCPServer compatibility"""
        role_input = kwargs.get("user_role", UserRole.SUPPORT)
        if isinstance(role_input, str):
            role = UserRole(role_input)
        else:
            role = role_input
            
        limit = kwargs.get("max_results", 10)
        status = kwargs.get("status")
        
        # Debug logging
        self.logger.info(f"search_documents called with query: '{query}', role: {role.value}")
        
        raw = await self.search_tickets(query, role.value, status=status, max_results=limit)
        if "error" in raw:
            self.logger.error(f"search_tickets returned error: {raw['error']}")
            if HTTPException:
                raise HTTPException(status_code=403, detail=raw["error"])
            else:
                return SearchResponse(results=[], total_found=0, search_time=0.0)
        
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
        """Retrieve specific record by ID"""
        return self.tickets.get(document_id)
