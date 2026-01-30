"""
Embedding Configuration for MCP System (FIXED)
- Correct semantic chunk sizing
- Proper BGE query + document instructions
- Schema-controlled metadata propagation
- MCP-aligned access control defaults
"""

import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from FlagEmbedding import FlagModel
from dotenv import load_dotenv

load_dotenv()

# =========================
# Chunking Configuration
# =========================

@dataclass
class ChunkingConfig:
    """
    Character-based chunking (NOT token-based)
    Tuned for BGE semantic retrieval
    """
    chunk_size: int = 600          # characters
    chunk_overlap: int = 150       # characters
    min_chunk_size: int = 100
    max_chunk_size: int = 1200
    separators: List[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ]

# =========================
# Embedding Configuration
# =========================

@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages:"
    document_instruction: str = "Represent this document for retrieval:"
    max_sequence_length: int = 512
    vector_size: int = 384

# =========================
# Metadata Schema
# =========================

@dataclass
class MetadataField:
    name: str
    data_type: str
    required: bool = True
    indexed: bool = True
    description: str = ""

class MetadataSchema:

    CORE_FIELDS = {
        "source_id", "title", "content", "source_type",
        "chunk_id", "chunk_index", "total_chunks"
    }

    TIMESTAMP_FIELDS = {
        "created_date", "last_updated", "indexed_date", "chunk_created"
    }

    ACCESS_FIELDS = {
        "access_level", "allowed_roles", "team", "owner", "classification"
    }

    CONTENT_FIELDS = {
        "content_length", "token_count", "language", "content_type"
    }

    DOCUMENTATION_FIELDS = {
        "category", "author", "version", "tags", "view_count", "rating", "status"
    }

    TICKETS_FIELDS = {
        "customer", "priority", "severity", "status",
        "assigned_to", "resolved_date",
        "customer_satisfaction", "time_to_resolution"
    }

    RUNBOOKS_FIELDS = {
        "category", "severity", "author", "team",
        "version", "tags", "execution_count",
        "success_rate", "estimated_duration", "dependencies"
    }

    @classmethod
    def allowed_fields(cls, source_type: str) -> set:
        base = (
            cls.CORE_FIELDS |
            cls.TIMESTAMP_FIELDS |
            cls.ACCESS_FIELDS |
            cls.CONTENT_FIELDS
        )

        if source_type == "documentation":
            return base | cls.DOCUMENTATION_FIELDS
        if source_type == "tickets":
            return base | cls.TICKETS_FIELDS
        if source_type == "runbooks":
            return base | cls.RUNBOOKS_FIELDS

        return base

# =========================
# Document Chunker
# =========================

class DocumentChunker:

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = document.get("content", "").strip()
        if not content:
            return []

        # =========================
        # SOURCE-TYPE AWARE CHUNKING
        # =========================

        source_type = document.get("source_type", "unknown")
        
        # 🎯 ENHANCE: Set source type for code-aware chunking
        self._current_source_type = source_type

        # 🚫 DO NOT chunk tickets
        if source_type == "tickets":
            # 🎯 ENHANCE: Combine title + content for better semantic search
            title = document.get("title", "").strip()
            enhanced_content = f"{title}. {content}" if title and content else (title or content)
            
            now = datetime.utcnow().isoformat()
            return [{
                "source_id": document.get("id"),
                "title": document.get("title"),
                "content": content,  # Keep original content for display
                "enhanced_content": enhanced_content,  # Use this for embedding
                "source_type": source_type,

                "chunk_id": f"{document.get('id')}_full",
                "chunk_index": 0,
                "total_chunks": 1,

                "created_date": document.get("created_date"),
                "last_updated": document.get("last_updated"),
                "indexed_date": now,
                "chunk_created": now,

                "content_length": len(enhanced_content),
                "token_count": self._estimate_tokens(enhanced_content),
                "language": "en",
                "content_type": "text",

                # MCP-safe defaults
                "access_level": document.get("access_level", "internal"),
                "allowed_roles": document.get("allowed_roles", ["support", "engineering"]),
                "team": document.get("team"),
                "owner": document.get("author") or document.get("assigned_to"),
                "classification": document.get("classification", "internal"),

                # preserve ticket metadata safely
                **{
                    k: v for k, v in document.items()
                    if k in MetadataSchema.allowed_fields(source_type)
                }
            }]

        source_type = document.get("source_type", "unknown")
        allowed_fields = MetadataSchema.allowed_fields(source_type)

        chunks = self._create_chunks(content)
        now = datetime.utcnow().isoformat()

        chunked_docs = []
        for idx, text in enumerate(chunks):
            if len(text) < self.config.min_chunk_size:
                continue

            chunk = {
                "source_id": document.get("id"),
                "title": document.get("title"),
                "content": text,
                "source_type": source_type,
                "chunk_id": f"{document.get('id')}_chunk_{idx:03d}",
                "chunk_index": idx,
                "total_chunks": len(chunks),

                "created_date": document.get("created_date"),
                "last_updated": document.get("last_updated"),
                "indexed_date": now,
                "chunk_created": now,

                "content_length": len(text),
                "token_count": self._estimate_tokens(text),
                "language": "en",
                "content_type": "text",

                # MCP-safe defaults
                "access_level": document.get("access_level", "internal"),
                "allowed_roles": document.get("allowed_roles", ["support", "engineering"]),
                "team": document.get("team"),
                "owner": document.get("author") or document.get("assigned_to"),
                "classification": document.get("classification", "internal"),
            }

            # Controlled metadata copy
            for k, v in document.items():
                if k in allowed_fields and k not in chunk:
                    chunk[k] = v

            chunked_docs.append(chunk)

        return chunked_docs

    def _create_chunks(self, text: str) -> List[str]:
        """Code-aware chunking that preserves blocks and structured content"""
        source_type = getattr(self, '_current_source_type', 'unknown')
        
        # 🎯 ENHANCE: Use code-aware chunking for documentation and runbooks
        if source_type in ['documentation', 'runbooks']:
            return self._create_code_aware_chunks(text)
        else:
            return self._create_standard_chunks(text)
    
    def _create_code_aware_chunks(self, text: str) -> List[str]:
        """Preserve code blocks, tables, and structured sections"""
        import re
        
        # Define patterns for structured content
        patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'```[\w]*\n[\s\S]*?```',  # Code blocks with language
            r'```bash\n[\s\S]*?```',  # Bash blocks
            r'```ini\n[\s\S]*?```',  # INI blocks
            r'```yaml\n[\s\S]*?```',  # YAML blocks
            r'```json\n[\s\S]*?```',  # JSON blocks
            r'```\n[\s\S]*?```',  # Code blocks without language
            r'```[\s\S]*?```',  # Fallback code blocks
            r'\|.*\|.*\|',  # Markdown tables
            r'^#{1,6}\s.*$',  # Headers
            r'^\s*[-*+]\s+.*$',  # List items
            r'^\s*\d+\.\s+.*$',  # Numbered lists
        ]
        
        # Find all structured blocks
        structured_blocks = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                structured_blocks.append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group(),
                    'type': 'code_block' if '```' in match.group() else 'structured'
                })
        
        # Sort by position
        structured_blocks.sort(key=lambda x: x['start'])
        
        # If no structured content, use standard chunking
        if not structured_blocks:
            return self._create_standard_chunks(text)
        
        # Create chunks that preserve structured blocks
        chunks = []
        pos = 0
        
        for block in structured_blocks:
            # Add text before the block
            if pos < block['start']:
                before_text = text[pos:block['start']].strip()
                if before_text:
                    before_chunks = self._create_standard_chunks(before_text)
                    chunks.extend(before_chunks)
            
            # Add the structured block as a whole chunk
            block_content = block['content'].strip()
            if block_content:
                chunks.append(block_content)
            
            pos = block['end']
        
        # Add remaining text
        if pos < len(text):
            remaining_text = text[pos:].strip()
            if remaining_text:
                remaining_chunks = self._create_standard_chunks(remaining_text)
                chunks.extend(remaining_chunks)
        
        return chunks
    
    def _create_standard_chunks(self, text: str) -> List[str]:
        """Standard chunking for non-structured content"""
        chunks = []
        pos = 0
        length = len(text)

        while pos < length:
            end = min(pos + self.config.chunk_size, length)
            if end < length:
                end = self._find_break(text, pos, end)

            chunk = text[pos:end].strip()
            # Enforce semantic size bounds
            if chunk:
                if len(chunk) > self.config.max_chunk_size:
                    chunk = chunk[:self.config.max_chunk_size]
                chunks.append(chunk)

            next_pos = max(
                end - self.config.chunk_overlap,
                pos + self.config.chunk_size // 2
            )

            if next_pos <= pos:
                break

            pos = next_pos

        return chunks

    def _find_break(self, text: str, start: int, end: int) -> int:
        window_start = start + int((end - start) * 0.7)
        for sep in self.config.separators:
            idx = text.rfind(sep, window_start, end)
            if idx != -1:
                return idx + len(sep)
        return end

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

# =========================
# Embedding Processor
# =========================

class EmbeddingProcessor:

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    def initialize_model(self):
        self._model = FlagModel(
            self.config.model_name,
            query_instruction_for_retrieval=self.config.query_instruction,
            document_instruction_for_retrieval=self.config.document_instruction,
            use_fp16=True
        )

        if self.config.device != "cpu":
            if torch.cuda.is_available():
                self._model.model = self._model.model.to(self.config.device)
            elif torch.backends.mps.is_available():
                self._model.model = self._model.model.to("mps")

        print(f"✅ BGE initialized: {self.config.model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self.initialize_model()

        vectors = self._model.encode(texts, batch_size=self.config.batch_size)

        if self.config.normalize_embeddings:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

# =========================
# Globals
# =========================

CHUNKING_CONFIG = ChunkingConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
_processor: Optional[EmbeddingProcessor] = None

def get_chunker() -> DocumentChunker:
    return DocumentChunker(CHUNKING_CONFIG)

def get_embedding_processor() -> EmbeddingProcessor:
    global _processor
    if _processor is None:
        _processor = EmbeddingProcessor(EMBEDDING_CONFIG)
    return _processor

def get_embeddings() -> EmbeddingProcessor:
    """
    Backward-compatible alias used by orchestrator
    """
    return get_embedding_processor()