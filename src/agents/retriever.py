"""
Retriever component for knowledge agent
Handles communication with MCP servers and result ranking
"""

import logging
import math
from typing import List, Dict, Any
from core.models import RetrievedChunk, UserRole, DataSourceType
from agents.mcp_client import ContextServerManager

# Cross-encoder reranker imports (optional - will fallback if not available)
try:
    from FlagEmbedding import FlagReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("FlagReranker not available - will use basic relevance scoring only")

logger = logging.getLogger(__name__)

class KnowledgeRetriever:
    """Manages document retrieval and ranking across multiple servers"""
    
    def __init__(self, server_manager: ContextServerManager):
        self.server_manager = server_manager
        self.reranker = None
        self._init_reranker()
    
    def _init_reranker(self):
        """Initialize cross-encoder reranker if available"""
        if RERANKER_AVAILABLE:
            try:
                # Use BGE reranker for better relevance
                self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
                logger.info("✅ BGE-reranker initialized for cross-encoder reranking")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
        else:
            logger.info("📊 Using basic relevance scoring (reranker not available)")
    
    def _rerank_chunks(self, chunks: List[RetrievedChunk], query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder for better relevance"""
        if not chunks:
            return chunks
        
        if not self.reranker or len(chunks) <= 1:
            # Fallback to basic scoring if reranker not available or only 1 chunk
            return chunks[:top_k]
        
        try:
            # Prepare pairs for reranker (already limited to candidates)
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Get reranker scores
            rerank_scores = self.reranker.compute_score(pairs)
            
            # Update chunks with reranker scores
            for i, chunk in enumerate(chunks):
                if i < len(rerank_scores):
                    # Store reranker score in metadata
                    if chunk.metadata is None:
                        chunk.metadata = {}
                    chunk.metadata["_rerank_score"] = float(rerank_scores[i])
            
            # 🎯 FIX: Blend reranker score with final score for optimal ranking
            for chunk in chunks:
                rerank_score = chunk.metadata.get("_rerank_score", 0.0)
                final_score = chunk.metadata.get("_final_score", 0.0)
                
                # 🎯 ENHANCE: Sigmoid normalization for smoother ranking
                # BGE reranker scores can vary widely - sigmoid provides stable normalization
                normalized_rerank = 1 / (1 + math.exp(-rerank_score))
                
                # Blend scores: prioritize reranker (70%) but keep multi-signal context (30%)
                blended_score = 0.7 * normalized_rerank + 0.3 * final_score
                chunk.metadata["_blended_score"] = blended_score
            
            # Sort by blended score (descending)
            chunks.sort(key=lambda x: x.metadata.get("_blended_score", 0.0), reverse=True)
            
            logger.info(f"🔄 Blended reranker + multi-signal scores → top {top_k}")
            return chunks[:top_k]  # Return top_k blended-score chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks[:top_k]  # Fallback to original order

    async def retrieve_from_server(
        self, 
        server_name: str, 
        query: str, 
        max_results: int,
        min_confidence: float,
        user_role: UserRole = UserRole.ADMIN,
        **filters
    ) -> List[RetrievedChunk]:
        """Retrieve documents from a specific server with filters"""
        try:
            server = self.server_manager.servers.get(server_name)
            if not server or not server.is_healthy:
                return []
            
            # Clean filters
            active_filters = {k: v for k, v in filters.items() if v is not None and v != "" and v != "null"}
            
            response_data = await self.server_manager.search_server(
                server_name,
                query,
                max_results=max_results,
                min_confidence=min_confidence,
                user_role=user_role,
                **active_filters
            )
            
            chunks = []
            results = response_data.get("results", [])
            for result in results:
                chunk = RetrievedChunk(
                    content=result.get("content", ""),
                    source_id=result.get("id") or result.get("source_id") or "unknown",
                    source_type=server.data_type,
                    confidence_score=result.get("score") or result.get("confidence_score") or 0.0,
                    metadata=result,
                    vector_id=result.get("id")
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving from {server_name}: {e}")
            return []

    def rank_chunks(
        self, 
        chunks: List[RetrievedChunk], 
        query: str,
        min_confidence: float
    ) -> List[RetrievedChunk]:
        """Multi-signal ranking with server-aware score normalization"""
        # Filter by minimum confidence first
        filtered_chunks = [
            chunk for chunk in chunks 
            if chunk.confidence_score >= min_confidence
        ]
        
        if not filtered_chunks:
            return []
        
        # 🎯 ENHANCE: Normalize scores per server before comparison
        normalized_chunks = self._normalize_scores_per_server(filtered_chunks)
        
        # Multi-signal ranking with normalized scores
        scored_chunks = []
        for chunk in normalized_chunks:
            final_score = self._calculate_comprehensive_score(chunk, query)
            # 🎯 FIX: Store final_score in metadata, don't mutate object
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["_final_score"] = final_score
            scored_chunks.append(chunk)
        
        # Sort by comprehensive score (descending) - use metadata-stored score
        scored_chunks.sort(key=lambda x: x.metadata.get("_final_score", 0.0), reverse=True)
        
        # 🎯 ENHANCE: Deduplicate across servers
        deduplicated_chunks = self._deduplicate_chunks(scored_chunks)
        
        # 🚀 ENHANCE: Apply cross-encoder reranking for final relevance boost
        # Rerank only top candidates - cross-encoders are expensive!
        RERANK_CANDIDATES = 20  # Only rerank top 20 candidates
        FINAL_K = min(10, len(deduplicated_chunks))  # Return top 10 or fewer
        
        reranked_chunks = self._rerank_chunks(
            deduplicated_chunks[:RERANK_CANDIDATES],
            query,
            top_k=FINAL_K
        )
        
        return reranked_chunks
    
    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Deduplicate chunks across servers, keeping highest-confidence instance"""
        seen_chunks = {}
        deduplicated = []
        
        for chunk in chunks:
            # Create deduplication keys
            chunk_id = getattr(chunk, 'chunk_id', None)
            source_id = chunk.source_id
            content_hash = self._hash_content(chunk.content)
            
            # 🎯 FIX: Use compound keys to avoid false positives
            # Never dedupe on source_id alone - same doc can have multiple versions/chunks
            dedup_keys = []
            
            # Primary: Exact chunk_id match (most specific)
            if chunk_id:
                dedup_keys.append(f"chunk_id:{chunk_id}")
            
            # Secondary: Compound key (source_id + content_hash)
            # This prevents false positives from reused source_ids
            compound_key = f"{source_id}:{content_hash}"
            dedup_keys.append(f"compound:{compound_key}")
            
            # Tertiary: Content-only hash (for exact duplicate content)
            dedup_keys.append(f"content:{content_hash}")
            
            # Check if this chunk is a duplicate
            is_duplicate = False
            best_key = None
            
            for key in dedup_keys:
                if key in seen_chunks:
                    # Found duplicate - keep the one with higher final_score
                    existing_chunk = seen_chunks[key]
                    current_score = chunk.metadata.get("_final_score", chunk.confidence_score)
                    existing_score = existing_chunk.metadata.get("_final_score", existing_chunk.confidence_score)
                    
                    if current_score > existing_score:
                        # Replace with higher-scoring chunk
                        seen_chunks[key] = chunk
                        # Remove old chunk from deduplicated list if present
                        if existing_chunk in deduplicated:
                            deduplicated.remove(existing_chunk)
                        deduplicated.append(chunk)
                        print(f"🔄 Replaced duplicate {key} (score: {existing_score:.3f} → {current_score:.3f})")
                    else:
                        print(f"🗑️  Skipped duplicate {key} (score: {current_score:.3f} ≤ {existing_score:.3f})")
                    
                    is_duplicate = True
                    best_key = key
                    break
            
            if not is_duplicate:
                # New unique chunk
                deduplicated.append(chunk)
                # Register all dedup keys for this chunk
                for key in dedup_keys:
                    if key not in seen_chunks:
                        seen_chunks[key] = chunk
                print(f"✅ Added unique chunk {source_id} (score: {chunk.metadata.get('_final_score', chunk.confidence_score):.3f})")
        
        print(f"📊 Deduplication: {len(chunks)} → {len(deduplicated)} chunks")
        return deduplicated
    
    def _hash_content(self, content: str) -> str:
        """Create content hash for deduplication"""
        import hashlib
        
        # Normalize content for hashing
        normalized_content = content.strip().lower()
        # Remove extra whitespace
        normalized_content = ' '.join(normalized_content.split())
        
        return hashlib.md5(normalized_content.encode()).hexdigest()[:16]
    
    def _normalize_scores_per_server(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Normalize confidence scores within each server to ensure fair comparison"""
        # Group chunks by server
        server_chunks = {}
        for chunk in chunks:
            server_name = chunk.source_type.value if hasattr(chunk.source_type, 'value') else str(chunk.source_type)
            if server_name not in server_chunks:
                server_chunks[server_name] = []
            server_chunks[server_name].append(chunk)
        
        normalized_chunks = []
        
        for server_name, server_chunk_list in server_chunks.items():
            if len(server_chunk_list) == 1:
                # Single chunk: no normalization needed, but scale to 0-1
                chunk = server_chunk_list[0]
                normalized_score = min(chunk.confidence_score, 1.0)
                # 🎯 FIX: Store in metadata, don't mutate object
                if chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["_norm_score"] = normalized_score
                normalized_chunks.append(chunk)
            else:
                # Multiple chunks: apply min-max normalization
                scores = [chunk.confidence_score for chunk in server_chunk_list]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                if score_range == 0:
                    # All scores are the same
                    for chunk in server_chunk_list:
                        # 🎯 FIX: Store in metadata, don't mutate object
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        chunk.metadata["_norm_score"] = 0.5  # Neutral score
                        normalized_chunks.append(chunk)
                else:
                    # Min-max normalization: (score - min) / (max - min)
                    for chunk in server_chunk_list:
                        normalized_score = (chunk.confidence_score - min_score) / score_range
                        # 🎯 FIX: Store in metadata, don't mutate object
                        if chunk.metadata is None:
                            chunk.metadata = {}
                        chunk.metadata["_norm_score"] = normalized_score
                        normalized_chunks.append(chunk)
        
        return normalized_chunks
    
    def _calculate_comprehensive_score(self, chunk: RetrievedChunk, query: str) -> float:
        """Calculate comprehensive score using multiple signals"""
        metadata = chunk.metadata or {}
        
        # Base vector similarity score (most important) - use normalized score from metadata
        vector_score = metadata.get("_norm_score", chunk.confidence_score)
        alpha = 0.5  # Weight for vector similarity
        
        # Signal 1: Freshness score (newer is better)
        freshness_score = self._calculate_freshness_score(metadata)
        beta = 0.2  # Weight for freshness
        
        # Signal 2: Quality score (content quality indicators)
        quality_score = self._calculate_quality_score(chunk, metadata)
        gamma = 0.15  # Weight for quality
        
        # Signal 3: Usage/engagement score (if available)
        usage_score = self._calculate_usage_score(metadata)
        delta = 0.1  # Weight for usage
        
        # Signal 4: Source authority score (documentation > tickets > runbooks)
        authority_score = self._calculate_authority_score(chunk.source_type)
        epsilon = 0.05  # Weight for source authority
        
        # Signal 5: Query relevance boost (term matching)
        relevance_boost = self._calculate_relevance_boost(chunk.content, query)
        zeta = 0.1  # Weight for term relevance
        
        # 🎯 FIX: Normalize weights to sum to 1.0
        total_weight = alpha + beta + gamma + delta + epsilon + zeta
        
        # Final weighted score with normalized weights
        final_score = (
            (alpha / total_weight) * vector_score +
            (beta / total_weight) * freshness_score +
            (gamma / total_weight) * quality_score +
            (delta / total_weight) * usage_score +
            (epsilon / total_weight) * authority_score +
            (zeta / total_weight) * relevance_boost
        )
        
        # Ensure score is between 0 and 1
        return min(max(final_score, 0.0), 1.0)
    
    def _calculate_freshness_score(self, metadata: dict) -> float:
        """Calculate freshness score (newer content gets higher score)"""
        from datetime import datetime, timedelta
        
        # Try different date fields
        date_fields = ['last_updated', 'created_date', 'indexed_date', 'chunk_created']
        content_date = None
        
        for field in date_fields:
            if field in metadata and metadata[field]:
                try:
                    content_date = datetime.fromisoformat(metadata[field].replace('Z', '+00:00'))
                    break
                except:
                    continue
        
        if not content_date:
            return 0.5  # Neutral score if no date
        
        # Calculate age in days
        age_days = (datetime.now() - content_date).days
        
        # Freshness scoring: newer is better
        if age_days <= 7:
            return 1.0  # Very fresh (within week)
        elif age_days <= 30:
            return 0.8  # Fresh (within month)
        elif age_days <= 90:
            return 0.6  # Recent (within quarter)
        elif age_days <= 365:
            return 0.4  # Older (within year)
        else:
            return 0.2  # Very old
    
    def _calculate_quality_score(self, chunk: RetrievedChunk, metadata: dict) -> float:
        """Calculate quality score based on content characteristics"""
        score = 0.5  # Base score
        
        content = chunk.content
        
        # Length-based quality (not too short, not too long)
        content_length = len(content)
        if 100 <= content_length <= 1000:
            score += 0.2  # Good length
        elif content_length > 1000:
            score += 0.1  # Comprehensive but might be verbose
        
        # Structure indicators
        if '```' in content:  # Contains code blocks
            score += 0.1
        if '|' in content and content.count('|') >= 4:  # Contains tables
            score += 0.1
        if any(header in content for header in ['# ', '## ', '### ']):  # Has headers
            score += 0.1
        
        # Metadata quality
        if metadata.get('title'):
            score += 0.05  # Has title
        if metadata.get('classification') == 'internal':
            score += 0.05  # Properly classified
        
        return min(score, 1.0)
    
    def _calculate_usage_score(self, metadata: dict) -> float:
        """Calculate usage/engagement score"""
        score = 0.5  # Base score
        
        # These fields might be added by usage tracking system
        usage_fields = ['usage_count', 'view_count', 'access_count', 'feedback_score']
        
        for field in usage_fields:
            if field in metadata and metadata[field]:
                if field.endswith('_score'):
                    # Normalize score fields (assuming 0-5 or 0-10 scale)
                    score += min(metadata[field] / 5.0, 0.5)
                else:
                    # Normalize count fields (logarithmic scaling)
                    count = metadata[field]
                    if count > 0:
                        score += min(math.log10(count + 1) / 3.0, 0.5)
        
        return min(score, 1.0)
    
    def _calculate_authority_score(self, source_type) -> float:
        """Calculate source authority score"""
        # 🎯 FIX: Use enum-based authority mapping to avoid string drift
        from core.models import DataSourceType
        
        # Documentation is most authoritative, then tickets, then runbooks
        authority_map = {
            DataSourceType.DOCUMENTATION: 1.0,  # Official docs
            DataSourceType.TICKETS: 0.8,       # Real customer issues
            DataSourceType.RUNBOOKS: 0.6       # Operational procedures
        }
        
        # Handle both enum and string inputs for backward compatibility
        if isinstance(source_type, DataSourceType):
            return authority_map.get(source_type, 0.5)
        else:
            # Backward compatibility for string inputs
            try:
                enum_type = DataSourceType(source_type)
                return authority_map.get(enum_type, 0.5)
            except ValueError:
                return 0.5  # Default for unknown source types
    
    def _calculate_relevance_boost(self, content: str, query: str) -> float:
        """Calculate term-based relevance boost"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Exact query match
        if query_lower in content_lower:
            return 1.0
        
        # Partial word matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if word in content_lower)
        
        if matches == 0:
            return 0.0
        elif matches == len(query_words):
            return 0.8
        else:
            return matches / len(query_words) * 0.6
