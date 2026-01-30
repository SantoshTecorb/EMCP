"""
Vector Database Namespace Configuration
Defines Qdrant collections for each data source
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class NamespaceConfig:
    """Configuration for a single namespace"""
    collection_name: str
    description: str
    vector_size: int
    distance_metric: str
    access_roles: list
    data_file: str

# Namespace definitions
VECTOR_NAMESPACES: Dict[str, NamespaceConfig] = {
    "documentation": NamespaceConfig(
        collection_name="mcp_documentation",
        description="Internal documentation, API guides, technical specifications",
        vector_size=384,  # BGE-small embedding dimension
        distance_metric="cosine",
        access_roles=["admin", "engineer"],
        data_file="data/store/documentation.json"
    ),
    
    "tickets": NamespaceConfig(
        collection_name="mcp_tickets",
        description="Support tickets, CRM data, customer interactions",
        vector_size=384,
        distance_metric="cosine",
        access_roles=["admin", "engineer", "support", "manager"],
        data_file="data/store/tickets.json"
    ),
    
    "runbooks": NamespaceConfig(
        collection_name="mcp_runbooks",
        description="Engineering procedures, deployment guides, incident response",
        vector_size=384,
        distance_metric="cosine",
        access_roles=["admin", "engineer", "manager"],
        data_file="data/store/runbooks.json"
    )
}

# Role-based namespace access mapping
ROLE_NAMESPACE_ACCESS: Dict[str, list] = {
    "admin": ["documentation", "tickets", "runbooks"],
    "engineer": ["documentation", "runbooks", "tickets"],
    "support": ["tickets"],
    "sales": ["tickets"],
    "manager": ["tickets", "runbooks"]
}

# Collection creation parameters
COLLECTION_PARAMS = {
    "vectors": {
        "size": 1024,
        "distance": "Cosine",
        "hnsw": {
            "m": 16,
            "ef_construct": 100,
            "full_scan_threshold": 10000
        }
    },
    "optimizers_config": {
        "default_segment_number": 2,
        "max_segment_size": 200000,
        "memmap_threshold": 50000
    }
}

# Search configuration
SEARCH_CONFIG = {
    "default_limit": 10,
    "default_threshold": 0.7,
    "max_limit": 100,
    "min_threshold": 0.1
}

# Batch processing configuration
BATCH_CONFIG = {
    "batch_size": 100,
    "max_retries": 3,
    "retry_delay": 1.0
}

def get_namespace_config(source_type: str) -> NamespaceConfig:
    """Get namespace configuration for a source type"""
    return VECTOR_NAMESPACES.get(source_type)

def get_accessible_namespaces(user_role: str) -> list:
    """Get list of namespaces accessible to a user role"""
    return ROLE_NAMESPACE_ACCESS.get(user_role, [])

def get_all_collection_names() -> list:
    """Get all collection names"""
    return [config.collection_name for config in VECTOR_NAMESPACES.values()]

def get_collection_name(source_type: str) -> str:
    """Get collection name for a source type"""
    config = VECTOR_NAMESPACES.get(source_type)
    return config.collection_name if config else None

def validate_namespace_access(user_role: str, source_type: str) -> bool:
    """Check if user role can access a specific namespace"""
    accessible_namespaces = get_accessible_namespaces(user_role)
    return source_type in accessible_namespaces
