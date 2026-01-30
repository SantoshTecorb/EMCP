#!/usr/bin/env python3
"""
Modular Indexing Pipeline for MCP System
Restructured for better maintainability and clarity.
Supports CLI arguments for granular control.
"""

import sys
import os
import json
import uuid
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from config.embedding_config import (
    get_chunker, get_embedding_processor,
    EMBEDDING_CONFIG, MetadataSchema
)
from config.metadata_tracker import get_metadata_tracker
from config.vector_namespaces import get_namespace_config, get_all_collection_names

# Metadata fields that should be excluded from dynamic payload mapping to prevent schema collisions
PROTECTED_METADATA_FIELDS = [
    "embedding", "source_id", "title", "content", "source_type",
    "chunk_id", "chunk_index", "total_chunks", "created_date", 
    "last_updated", "indexed_date", "chunk_created", "content_length",
    "token_count", "language", "content_type", "source_system",
    "source_path", "source_version", "ingestion_batch", "processing_status",
    "access_level", "allowed_roles", "owner", "team", "classification",
    "quality_score", "freshness_score", "usage_count", "feedback_score", "last_accessed"
]

class SourceDataManager:
    """Handles loading and tracking of source data from various providers"""
    
    def __init__(self, metadata_tracker):
        self.metadata_tracker = metadata_tracker
        self.data_source_registry = {
            "documentation": "data/store/documentation.json",
            "tickets": "data/store/tickets.json",
            "runbooks": "data/store/runbooks.json"
        }

    def load_all_source_data(self) -> Dict[str, List[Dict]]:
        """Loads and prepares data from all registered sources"""
        source_data_by_type = {}
        
        for source_type, file_path in self.data_source_registry.items():
            absolute_path = Path(file_path)
            if not absolute_path.exists():
                print(f"   Source file not found: {file_path}")
                continue
            
            with open(absolute_path, 'r', encoding='utf-8') as source_file:
                raw_records = json.load(source_file)
                source_data_by_type[source_type] = raw_records
                print(f"   Loaded {len(raw_records)} records from {source_type}")
                
                for record in raw_records:
                    self.metadata_tracker.track_source(
                        source_id=record.get("id"),
                        source_system="mcp_system",
                        source_path=file_path,
                        source_version="1.0"
                    )
        return source_data_by_type

class VectorStoreManager:
    """Handles interactions with the Qdrant vector database"""
    
    def __init__(self, connection_url: str = "http://localhost:6333"):
        self.db_client = QdrantClient(connection_url)

    def synchronize_collections(self, vector_dimension: int):
        """Ensures all required collections exist and match the current schema"""
        print("Synchronizing vector collections...")
        for collection_name in get_all_collection_names():
            try:
                self.db_client.delete_collection(collection_name)
                print(f"   🗑️  Reset existing collection: {collection_name}")
            except Exception:
                # Collection might not exist, which is fine
                pass
            
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": vector_dimension,
                    "distance": "Cosine"
                }
            )
            print(f"   Initialized collection: {collection_name}")

    def upload_points_in_batches(self, collection_name: str, points: List[PointStruct], batch_size: int = 100):
        """Uploads vector points to Qdrant using specialized batching logic"""
        for i in range(0, len(points), batch_size):
            current_batch = points[i:i + batch_size]
            self.db_client.upsert(collection_name=collection_name, points=current_batch)

class DocumentationManager:
    """Handles generation of detailed indexing manifests and audit reports"""
    
    @staticmethod
    def generate_indexing_report(pipeline_stats: Dict[str, Any], report_output_path: str):
        """Generates a comprehensive JSON report of the indexing execution"""
        indexing_manifest = {
            "metadata": {
                "pipeline_version": "2.1",
                "execution_timestamp": datetime.now().isoformat(),
                "description": "Production-grade MCP indexing audit trail",
            },
            "performance_metrics": pipeline_stats,
            "orchestrated_components": [
                "SourceDataManager",
                "DocumentChunker",
                "EmbeddingProcessor",
                "VectorStoreManager"
            ]
        }
        
        output_file = Path(report_output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(indexing_manifest, f, indent=2)
            
        print(f"Audit report generated: {report_output_path}")

class IndexingPipeline:
    """The main orchestrator for the MCP document indexing lifecycle"""
    
    def __init__(self, skip_ingestion: bool = False):
        self.chunker = get_chunker()
        self.embedding_engine = get_embedding_processor()
        self.metadata_tracker = get_metadata_tracker()
        self.vector_store_manager = VectorStoreManager()
        self.skip_ingestion = skip_ingestion
        
        self.execution_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "vectors_uploaded": 0,
            "start_timestamp": time.time()
        }

    def execute_pipeline(self):
        """Starts the end-to-end indexing process"""
        print("Executing MCP Production Indexing Pipeline")
        print("=" * 60)

        if not self.skip_ingestion:
            # Phase 1: Data Ingestion
            data_manager = SourceDataManager(self.metadata_tracker)
            source_data_by_type = data_manager.load_all_source_data()

            # Phase 2: Document Chunking
            print("\nPhase 2: Structural Document Chunking")
            processed_chunks_by_source = {}
            for source_type, raw_records in source_data_by_type.items():
                source_specific_chunks = []

                for record in raw_records:
                    if source_type == "tickets":
                        ticket_id = record.get("id")  # from your JSON
                        record["source_id"] = ticket_id  # ✅ ensure source_id is set
                        record["chunk_id"] = f"{ticket_id}_full"
                        record["chunk_index"] = 0
                        record["total_chunks"] = 1
                        source_specific_chunks.append(record)
                    else:
                        source_specific_chunks.extend(self.chunker.chunk_document(record))

                processed_chunks_by_source[source_type] = source_specific_chunks
                self.execution_stats["documents_processed"] += len(raw_records)
                self.execution_stats["chunks_created"] += len(source_specific_chunks)

                print(
                    f"   {source_type}: Transmuted into "
                    f"{len(source_specific_chunks)} semantic units"
                )
                processed_chunks_by_source[source_type] = source_specific_chunks
                self.execution_stats["documents_processed"] += len(raw_records)
                self.execution_stats["chunks_created"] += len(source_specific_chunks)
                print(f"   {source_type}: Transmuted into {len(source_specific_chunks)} semantic chunks")

            # Phase 3: Embedding Synthesis
            print("\nPhase 3: Neural Embedding Synthesis")
            for source_type, document_chunks in processed_chunks_by_source.items():
                print(f"   Encoding {source_type} stream...")
                # 🎯 ENHANCE: Use enhanced_content for tickets (title + content)
                if source_type == "tickets":
                    chunk_contents = [chunk.get("enhanced_content", chunk["content"]) for chunk in document_chunks]
                else:
                    chunk_contents = [chunk["content"] for chunk in document_chunks]
                embedding_vectors = self.embedding_engine.embed_texts(chunk_contents)
                
                for chunk, vector in zip(document_chunks, embedding_vectors):
                    chunk["embedding_vector"] = vector
                
                self.execution_stats["embeddings_generated"] += len(embedding_vectors)

            # Phase 4: Vector Store Persistence
            print("\nPhase 4: Vector Matrix Persistence")
            self.vector_store_manager.synchronize_collections(EMBEDDING_CONFIG.vector_size)
            
            for source_type, document_chunks in processed_chunks_by_source.items():
                collection_config = get_namespace_config(source_type)
                target_collection = collection_config.collection_name
                qdrant_points = []
                
                for chunk in document_chunks:
                    # Construct sanitized metadata payload
                    metadata_payload = {k: v for k, v in chunk.items() if k not in PROTECTED_METADATA_FIELDS}
                    metadata_payload.update({
                        "source_id": chunk.get("source_id"),
                        "title": chunk.get("title"),
                        "content": chunk.get("content"),
                        "source_type": chunk.get("source_type"),
                        "chunk_id": chunk.get("chunk_id"),
                        "chunk_index": chunk.get("chunk_index"),
                        "total_chunks": chunk.get("total_chunks"),
                        "indexed_at": datetime.now().isoformat()
                    })
                    
                    point_id = str(uuid.uuid4())
                    qdrant_points.append(PointStruct(
                        id=point_id,
                        vector=chunk["embedding_vector"],
                        payload=metadata_payload
                    ))
                
                self.vector_store_manager.upload_points_in_batches(target_collection, qdrant_points)
                self.execution_stats["vectors_uploaded"] += len(qdrant_points)
                print(f"   {source_type}: Indexed and persisted to {target_collection}")
        else:
            print("\nIngestion phase bypassed via configuration.")

        # Phase 5: Semantic Integrity Verification
        self._verify_system_integrity()

        # Phase 6: Audit Documentation
        if not self.skip_ingestion:
            self.execution_stats["total_latency_seconds"] = time.time() - self.execution_stats["start_timestamp"]
            DocumentationManager.generate_indexing_report(self.execution_stats, "docs/indexing_audit_log.json")

    def _verify_system_integrity(self):
        """Performs high-precision semantic search probes to verify system health"""
        print("\n🔍 Phase 5: Automated Semantic Integrity Probing")
        integrity_check_queries = [
            ("API authentication protocols", "documentation"),
            ("critical redundant connection timeout", "tickets"),
            ("production deployment workflows", "runbooks")
        ]
        
        for probe_query, target_namespace in integrity_check_queries:
            try:
                namespace_config = get_namespace_config(target_namespace)
                target_collection = namespace_config.collection_name
                
                # Synthesize probe vector
                probe_vector = self.embedding_engine.embed_query(probe_query)
                
                # Execute semantic discovery
                discovery_results = self.vector_store_manager.db_client.query_points(
                    collection_name=target_collection,
                    query=probe_vector,
                    limit=1
                ).points
                
                if discovery_results:
                    top_match = discovery_results[0]
                    match_title = top_match.payload.get('title', 'Unknown Title')
                    print(f"   Probe '{probe_query}' [Namespace: {target_namespace}]: ✅ Integrity Confirmed")
                    print(f"      Top Relevance: {match_title} | Confidence: {top_match.score:.4f}")
                else:
                    print(f"   Probe '{probe_query}' [Namespace: {target_namespace}]: ⚠️ Signal Missing")
            except Exception as pipeline_error:
                print(f"   Probe Failure [Namespace: {target_namespace}]: {pipeline_error}")

if __name__ == "__main__":
    cli_orchestrator = argparse.ArgumentParser(description="MCP Enterprise Indexing Suite")
    cli_orchestrator.add_argument(
        "--skip-ingestion", 
        action="store_true", 
        help="Bypasses data ingestion and indexing phases, executing only integrity probes."
    )
    runtime_args = cli_orchestrator.parse_args()
    
    pipeline_orchestrator = IndexingPipeline(skip_ingestion=runtime_args.skip_ingestion)
    pipeline_orchestrator.execute_pipeline()
