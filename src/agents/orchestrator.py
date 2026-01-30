"""
Orchestrator component for enterprise knowledge agent
Coordinates the flow across parsing, retrieval, and generation
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_groq import ChatGroq

from core.models import (
    QueryRequest, QueryResponse, UserRole, DataSourceType, ContextServer, 
    Citation, UsageStats, RetrievedChunk
)
from agents.mcp_client import ContextServerManager
from core.permissions import PermissionManager
from agents.query_parser import QueryIntentParser
from agents.retriever import KnowledgeRetriever
from agents.generator import KnowledgeGenerator
from agents.planner_agent import PlannerAgent
from core.prompts import PromptManager
from config.embedding_config import get_embeddings
from core.audit_logger import AuditLogger
from config.cost_config import calculate_cost

logger = logging.getLogger(__name__)

class KnowledgeOrchestrator:
    """Main orchestrator for query processing"""
    
    def __init__(
        self,
        llm_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.llm = ChatGroq(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.server_manager = ContextServerManager()
        self.permission_manager = PermissionManager()
        self.prompt_manager = PromptManager()
        self.intent_parser = QueryIntentParser()
        self.embeddings = get_embeddings()
        
        # Modular components
        self.retriever = KnowledgeRetriever(self.server_manager)
        
        self.generator = KnowledgeGenerator(self.llm)
        self.audit_logger = AuditLogger()
        self.planner_agent = PlannerAgent()
        self._register_default_servers()

    def _is_greeting(self, text: str) -> bool:
        """Simple deterministic check for greetings (Pre-routing layer)"""
        greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
        clean_text = text.lower().strip().strip("!").strip("?").strip(".")
        # 🎯 FIX: Check if text starts with any greeting (more flexible)
        return any(clean_text.startswith(g) for g in greetings)

    async def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        
        if isinstance(request.user_role, str):
            request.user_role = UserRole(request.user_role)

        # 1. Pre-routing Layer: Handle greetings without invoking RAG
        if self._is_greeting(request.question):
            return QueryResponse(
                answer="Hello! I am your Enterprise Knowledge Agent. How can I help you today?",
                citations=[],
                confidence_score=1.0,
                sources_used=[],
                processing_time=time.time() - start_time
            )

        try:
            print(f"\n=== ORCHESTRATOR QUERY START ===")
            print(f"Question: {request.question}")
            print(f"User Role: {request.user_role}")
            
            # 1. Planner Agent Analysis
            plan = await self.planner_agent.create_comprehensive_plan(
                request.question, 
                request.user_role
            )
            
            print(f"\n=== PLANNER OUTPUT ===")
            print(f"Task type: {plan['task_type']}")
            print(f"Recommended sources: {[s.value for s in plan['recommended_sources']]}")
            
            servers_to_search = [source.value for source in plan['recommended_sources']]
            print(f"Servers to search: {servers_to_search}")
            
            if not servers_to_search:
                return QueryResponse(
                    answer="You do not have permission to access any available data sources.",
                    citations=[],
                    confidence_score=0.0,
                    sources_used=[],
                    processing_time=time.time() - start_time,
                    usage=UsageStats()
                )
            
            # 2. Parallel Retrieval from Multiple Servers
            print(f"\n=== RETRIEVAL START ===")
            retrieval_tasks = []
            for server_name in servers_to_search:
                print(f"Creating retrieval task for {server_name}")
                task = asyncio.create_task(
                    self.retriever.retrieve_from_server(
                        server_name=server_name,
                        query=request.question,
                        max_results=request.max_results,  # 🎯 FIX: Use request's max_results
                        min_confidence=0.1
                    )
                )
                retrieval_tasks.append((task, server_name))
            
            print(f"Waiting for {len(retrieval_tasks)} retrieval tasks...")
            retrieval_results = await asyncio.gather(*[task for task, _ in retrieval_tasks], return_exceptions=True)
            print(f"Retrieval completed. Got {len(retrieval_results)} results")
            
            # 3. Server-Specific Agents Process Chunks
            all_chunks = []
            sources_used = []
            
            for (task, server_name), result in zip(retrieval_tasks, retrieval_results):
                print(f"\n--- Processing {server_name} ---")
                print(f"Result type: {type(result)}")
                
                if isinstance(result, Exception):
                    print(f"ERROR retrieving from {server_name}: {result}")
                    continue
                
                # FIX: Handle List[RetrievedChunk] returned by KnowledgeRetriever
                if isinstance(result, list):
                    print(f"✓ Received {len(result)} RetrievedChunk objects")
                    
                    if result:
                        # 🎯 FIX: Collect all raw chunks first, then use sophisticated ranking
                        all_chunks.extend(result)
                        sources_used.append(server_name)
                        print(f"✓ Collected {len(result)} raw chunks from {server_name}")
                    else:
                        print(f"✗ Empty result list")
                        
                else:
                    print(f"✗ Unexpected format: {type(result)}")
            
            print(f"\n=== CHUNKS SUMMARY ===")
            print(f"Total raw chunks collected: {len(all_chunks)}")
            print(f"Sources used: {sources_used}")
            
            # 🎯 ENHANCE: Apply sophisticated ranking pipeline to all collected chunks
            print(f"\n=== SOPHISTICATED RANKING ===")
            print(f"Applying multi-signal ranking, deduplication, and cross-encoder reranking...")
            
            ranked_chunks = self.retriever.rank_chunks(
                all_chunks,
                query=request.question,
                min_confidence=0.3
            )
            
            print(f"✅ Ranked {len(ranked_chunks)} chunks with full pipeline")
            
            for i, chunk in enumerate(ranked_chunks[:5]):  # Show top 5 ranked chunks
                final_score = chunk.metadata.get("_blended_score", chunk.metadata.get("_final_score", 0.0))
                rerank_score = chunk.metadata.get("_rerank_score", 0.0)
                print(f"Chunk {i}: source={chunk.source_id}, final={final_score:.3f}, rerank={rerank_score:.3f}")
                print(f"  Content: {chunk.content[:100]}...")
            
            # 4. Generate Answer with ROI tracking
            print(f"\n=== ANSWER GENERATION ===")
            print(f"Generating answer with {len(ranked_chunks)} ranked chunks")
            
            if not ranked_chunks:
                print("ERROR: No chunks available for answer generation!")
                # Return early with fallback
                return QueryResponse(
                    answer="I couldn't find relevant information to answer your question.",
                    citations=[],
                    confidence_score=0.0,
                    sources_used=[],
                    processing_time=time.time() - start_time,
                    usage=UsageStats()
                )
            
            system_instructions = self.prompt_manager.get_prompt("answer_generation", request.user_role.value)
            answer, citations, gen_usage = await self.generator.generate_answer(
                question=request.question,
                chunks=ranked_chunks,  # FIX: Use ranked chunks instead of raw chunks
                system_instructions=system_instructions,  # 🎯 FIX: Correct parameter name
                history=request.history
            )
            
            print(f"\n=== ANSWER GENERATED ===")
            print(f"Answer length: {len(answer)} chars")
            print(f"Answer preview: {answer[:200]}...")
            print(f"Citations count: {len(citations)}")
            
                
            # Total Usage Aggregation (no parse_usage in new architecture)
            total_usage = UsageStats(
                prompt_tokens=gen_usage.prompt_tokens,
                completion_tokens=gen_usage.completion_tokens,
                total_tokens=gen_usage.total_tokens
            )
            
            # Calculate total cost
            total_usage.estimated_cost_usd = calculate_cost(
                self.llm.model_name, 
                total_usage.prompt_tokens, 
                total_usage.completion_tokens
            )
            
            response = QueryResponse(
                answer=answer,
                citations=citations,
                confidence_score=self.generator.calculate_confidence(ranked_chunks, reranker_used=True),
                sources_used=[c.source_id for c in citations],
                fallback_used=len(all_chunks) == 0,
                processing_time=time.time() - start_time,
                usage=total_usage
            )
            
            # Log for ROI Auditing
            self.audit_logger.log_interaction(
                question=request.question,
                role=request.user_role.value,
                response_data=response.model_dump()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return QueryResponse(
                answer="Error processing request.",
                citations=[],
                confidence_score=0.0,
                sources_used=[],
                fallback_used=True,
                processing_time=time.time() - start_time
            )

    def _register_default_servers(self):
        """Register context servers"""
        default_servers = [
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
        for s in default_servers:
            self.server_manager.register_server(s)