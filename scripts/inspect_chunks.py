"""
Inspect retrieved chunks for a query (NO generation)
MCP-safe, retrieval-only debugging utility
"""

import sys
import os
import asyncio
from datetime import datetime

# =========================
# FIX: Add project root to PYTHONPATH
# =========================
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = PROJECT_ROOT  # MCP root contains core/, agents/, config/

sys.path = [SRC_ROOT] + sys.path
# =========================
# MCP imports (now work)
# =========================
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.models import RetrievedChunk, DataSourceType, ContextServer, UserRole
from agents.mcp_client import ContextServerManager
from agents.retriever import KnowledgeRetriever


async def inspect_query_chunks(query: str, server_name: str = "documentation"):
    print(f"\n🔍 QUERY: {query}")
    print(f"📡 SERVER: {server_name}\n")

    server_manager = ContextServerManager()
    retriever = KnowledgeRetriever(server_manager)

    # Same servers as orchestrator
    servers = [
        ContextServer(
            name="documentation",
            url="http://localhost:8001",
            data_type=DataSourceType.DOCUMENTATION,
            allowed_roles=[UserRole.ADMIN, UserRole.ENGINEER, UserRole.SUPPORT],
            last_check=datetime.now()
        ),
        ContextServer(
            name="tickets",
            url="http://localhost:8002",
            data_type=DataSourceType.TICKETS,
            allowed_roles=[UserRole.ADMIN, UserRole.SUPPORT, UserRole.MANAGER],
            last_check=datetime.now()
        ),
        ContextServer(
            name="runbooks",
            url="http://localhost:8003",
            data_type=DataSourceType.RUNBOOKS,
            allowed_roles=[UserRole.ADMIN, UserRole.ENGINEER],
            last_check=datetime.now()
        )
    ]

    for s in servers:
        server_manager.register_server(s)

    results = await retriever.retrieve_from_server(
        server_name=server_name,
        query=query,
        max_results=10,
        min_confidence=0.0  # inspect everything
    )

    if not results:
        print("❌ No chunks returned")
        return

    print(f"✅ Retrieved {len(results)} chunks\n")

    for i, chunk in enumerate(results):
        print(f"--- CHUNK {i + 1} ---")
        print(f"Source ID   : {chunk.source_id}")
        print(f"Vector ID   : {chunk.vector_id}")
        print(f"Confidence  : {chunk.confidence_score:.4f}")
        print(f"Source Type : {chunk.source_type}")
        print("Content:")
        print(chunk.content[:500])
        print("-" * 80)

    scores = [c.confidence_score for c in results]
    print("\n📊 Score distribution:", [round(s, 3) for s in scores])
    print("🎯 Inspection complete")


if __name__ == "__main__":
    query = input("Enter query: ").strip()
    server = input("Server (documentation/tickets/runbooks) [documentation]: ").strip() or "documentation"
    asyncio.run(inspect_query_chunks(query, server))