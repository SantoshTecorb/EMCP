"""
Enterprise Knowledge Agent
Refactored to use modular Orchestrator, Retriever, and Generator
"""

import logging
from typing import List, Dict, Any, Optional
from core.models import QueryRequest, QueryResponse
from agents.orchestrator import KnowledgeOrchestrator

logger = logging.getLogger(__name__)

class EnterpriseKnowledgeAgent:
    """Backward-compatible wrapper for KnowledgeOrchestrator"""
    
    def __init__(
        self,
        llm_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.orchestrator = KnowledgeOrchestrator(
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Expose internal components for backward compatibility if needed
        self.llm = self.orchestrator.llm
        self.server_manager = self.orchestrator.server_manager
        self.permission_manager = self.orchestrator.permission_manager
        self.intent_parser = self.orchestrator.intent_parser
        self.embeddings = self.orchestrator.embeddings

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Forward query to orchestrator"""
        return await self.orchestrator.query(request)
    
    async def compare_contexts(
        self, 
        question: str, 
        user_role: str,
        source_types: List[str]
    ) -> Dict[str, Any]:
        """Comparison logic - specialized legacy method"""
        # This could be moved to its own comparison module if needed
        comparison = {}
        for source_type in source_types:
            chunks = await self.orchestrator.retriever.retrieve_from_server(
                source_type, question, 5, 0.5
            )
            if chunks:
                comparison[source_type] = {
                    "document_count": len(chunks),
                    "avg_confidence": sum(c.confidence_score for c in chunks) / len(chunks),
                    "top_chunks": [
                        {"content": c.content[:200], "confidence": c.confidence_score, "source_id": c.source_id}
                        for c in chunks[:3]
                    ]
                }
            else:
                comparison[source_type] = {"document_count": 0, "avg_confidence": 0.0, "top_chunks": []}
        return comparison
