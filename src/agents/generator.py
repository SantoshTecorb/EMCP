"""
Generator component for knowledge agent
Handles LLM interactions and response synthesis
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.models import Citation, RetrievedChunk, ChatMessage, UsageStats

logger = logging.getLogger(__name__)

class KnowledgeGenerator:
    """Manages response generation and citation synthesis"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    async def generate_answer(
        self, 
        question: str, 
        chunks: List[RetrievedChunk],
        system_instructions: str,
        history: Optional[List[ChatMessage]] = None
    ) -> tuple[str, List[Citation], UsageStats]:
        """Generate answer with citations using LLM"""
        if not chunks:
            return "I couldn't find relevant information to answer your question.", [], UsageStats()
        
        # 1. Context Window Safeguard (Basic capping)
        # We cap at top 5 chunks and truncate individual chunks if they are extreme
        safe_chunks = chunks[:5] 
        
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(safe_chunks):
            # Limit each chunk to ~1500 chars (~375 tokens) to protect context window
            content = chunk.content[:1500]
            context_parts.append(f"[Source {i+1}] {content}")
            
        context = "\n\n".join(context_parts)
        
        msg_list = [SystemMessage(content=system_instructions)]
        
        # 🛡️ SAFETY: Handle conversation history safely
        if history:
            msg_list.append(
                SystemMessage(
                    content="Conversation history is provided for context only. "
                            "Do not use it as a knowledge source."
                )
            )
        
        # 2. Add history (Safe handling)
        if history:
            for msg in history:
                if msg.role == "user":
                    msg_list.append(HumanMessage(content=msg.content))
                else:
                    msg_list.append(AIMessage(content=msg.content))
        
        # Add context and question
        # Enforce grounding in the prompt
        context_msg = (
            f"CONTEXT FROM INTERNAL SERVERS:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "INSTRUCTIONS: Answer the question using ONLY the context provided above. "
            "If the answer is not in the context, state that you do not know. "
            "Use citations like [Source 1], [Source 2] where appropriate."
        )
        msg_list.append(HumanMessage(content=context_msg))
        
        response = await self.llm.ainvoke(msg_list)
        answer = response.content
        
        # Extract usage metadata
        metadata = response.response_metadata if hasattr(response, 'response_metadata') else {}
        token_usage = metadata.get("token_usage", {})
        
        usage = UsageStats(
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0)
        )
        
        # 🎯 ENHANCE: Only include citations that are actually referenced in the answer
        used_sources = set()
        for i in range(len(safe_chunks)):
            if f"[Source {i+1}]" in answer:
                used_sources.add(i)
        
        # Create citations only for referenced sources
        citations = []
        for i in used_sources:
            chunk = safe_chunks[i]
            citation = Citation(
                source_id=chunk.source_id,
                source_name=f"{chunk.source_type.value.title()} - {chunk.source_id}",
                content_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                confidence_score=chunk.confidence_score,
                metadata=chunk.metadata,
                last_updated=datetime.now()
            )
            citations.append(citation)
        
        return answer, citations, usage

    def calculate_confidence(
        self,
        chunks: List[RetrievedChunk],
        reranker_used: bool = False
    ) -> float:
        """Calculate overall confidence score with reranker calibration and safety penalties"""
        if not chunks:
            return 0.0
        
        # 🎯 ENHANCE: Calculate average confidence with proper calibration
        avg = sum(c.confidence_score for c in chunks) / len(chunks)
        
        # Apply reranker calibration penalty
        if reranker_used:
            avg *= 0.95  # calibration penalty for reranker scores
        
        # Apply safety penalty for low source volume
        if len(chunks) < 2:
            avg *= 0.7
        
        return round(min(1.0, avg), 2)
