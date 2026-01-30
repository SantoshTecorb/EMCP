"""
Data models for the MCP system
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    SUPPORT = "support"
    SALES = "sales"
    MANAGER = "manager"
    HR = "hr"


class UsageStats(BaseModel):
    """Token usage and cost metrics"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class DataSourceType(str, Enum):
    DOCUMENTATION = "documentation"
    TICKETS = "tickets"
    RUNBOOKS = "runbooks"


class QueryIntent(BaseModel):
    """Extracted intent and metadata from a natural language query"""
    primary_source: Optional[DataSourceType] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    entities: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    task_type: Optional[str] = None  # 🎯 ENHANCE: Add task_type for intelligent routing


class UnauthorizedAccessException(Exception):
    """Raised when a user attempts to access a data source they don't have permission for"""
    def __init__(self, role: UserRole, source_type: DataSourceType):
        self.role = role
        self.source_type = source_type
        super().__init__(f"Role '{role.value}' is not authorized to access '{source_type.value}' source.")


class Citation(BaseModel):
    source_id: str
    source_name: str
    content_snippet: str
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime


class RetrievedChunk(BaseModel):
    content: str
    source_id: str
    source_type: DataSourceType
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    user_role: UserRole
    history: Optional[List[ChatMessage]] = None
    max_results: int = Field(default=10)
    min_confidence: float = Field(default=0.4)
    include_sources: List[DataSourceType] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float
    sources_used: List[str]
    fallback_used: bool = False
    processing_time: float
    usage: UsageStats = Field(default_factory=UsageStats)


class ContextServer(BaseModel):
    name: str
    url: str
    data_type: DataSourceType
    allowed_roles: List[UserRole]
    is_healthy: bool = True
    last_check: datetime
    document_count: int = 0
    freshness_score: float = 1.0


class UserPermissions(BaseModel):
    user_id: str
    role: UserRole
    allowed_sources: List[DataSourceType]
    custom_restrictions: Dict[str, Any] = Field(default_factory=dict)
