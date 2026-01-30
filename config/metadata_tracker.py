"""
Metadata tracking and management system for MCP (FIXED)
- Safe datetime persistence
- Schema-aligned validation
- Deterministic metadata enrichment
- MCP-compatible access & quality signals
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field

# =====================================================
# Utilities
# =====================================================

def now_utc() -> datetime:
    return datetime.utcnow()

def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

def iso_to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

# =====================================================
# Source Tracking
# =====================================================

@dataclass
class SourceTracking:
    source_id: str
    source_system: str
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    source_version: Optional[str] = None
    ingestion_batch: str = ""
    processing_status: str = "pending"
    last_sync: Optional[datetime] = None
    sync_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.ingestion_batch:
            self.ingestion_batch = (
                f"batch_{now_utc().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["last_sync"] = dt_to_iso(self.last_sync)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceTracking":
        data = data.copy()
        data["last_sync"] = iso_to_dt(data.get("last_sync"))
        return cls(**data)

# =====================================================
# Timestamp Tracking
# =====================================================

@dataclass
class TimestampTracking:
    created_date: datetime
    last_updated: datetime
    indexed_date: datetime
    chunk_created: datetime
    last_accessed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    expiry_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "created_date": dt_to_iso(self.created_date),
            "last_updated": dt_to_iso(self.last_updated),
            "indexed_date": dt_to_iso(self.indexed_date),
            "chunk_created": dt_to_iso(self.chunk_created),
            "last_accessed": dt_to_iso(self.last_accessed),
            "next_review": dt_to_iso(self.next_review),
            "expiry_date": dt_to_iso(self.expiry_date),
        }

# =====================================================
# Metadata Field Manager
# =====================================================

class MetadataFieldManager:
    """Schema-controlled metadata validation"""

    def __init__(self):
        self.field_definitions = self._load_field_definitions()

    def _load_field_definitions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        from config.embedding_config import MetadataSchema

        definitions = {}
        for source_type in ["documentation", "tickets", "runbooks"]:
            allowed = MetadataSchema.allowed_fields(source_type)
            definitions[source_type] = {
                field: {
                    "required": field in MetadataSchema.CORE_FIELDS,
                }
                for field in allowed
            }
        return definitions

    def validate_metadata(
        self, source_type: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        schema = self.field_definitions.get(source_type)
        if not schema:
            return {"valid": False, "errors": ["Unknown source_type"], "warnings": []}

        errors, warnings = [], []

        # Unknown fields
        for field in metadata:
            if field not in schema:
                warnings.append(f"Unknown field ignored: {field}")

        # Required fields
        for field, spec in schema.items():
            if spec["required"] and field not in metadata:
                errors.append(f"Missing required field: {field}")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

# =====================================================
# Metadata Tracker
# =====================================================

class MetadataTracker:
    """Central MCP metadata authority"""

    def __init__(self, data_dir: str = "data/metadata"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.field_manager = MetadataFieldManager()
        self.source_tracking: Dict[str, SourceTracking] = {}
        self.timestamp_tracking: Dict[str, Dict[str, str]] = {}

        self._load()

    # ---------------------------
    # Persistence
    # ---------------------------

    def _load(self):
        source_file = self.data_dir / "source_tracking.json"
        if source_file.exists():
            with open(source_file) as f:
                raw = json.load(f)
                for sid, data in raw.items():
                    self.source_tracking[sid] = SourceTracking.from_dict(data)

        ts_file = self.data_dir / "timestamp_tracking.json"
        if ts_file.exists():
            with open(ts_file) as f:
                self.timestamp_tracking = json.load(f)

    def _save(self):
        with open(self.data_dir / "source_tracking.json", "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.source_tracking.items()},
                f,
                indent=2,
            )

        with open(self.data_dir / "timestamp_tracking.json", "w") as f:
            json.dump(self.timestamp_tracking, f, indent=2)

    # ---------------------------
    # Source Tracking
    # ---------------------------

    def track_source(self, source_id: str, source_system: str, **kwargs) -> SourceTracking:
        tracking = SourceTracking(source_id, source_system, **kwargs)
        self.source_tracking[source_id] = tracking
        self._save()
        return tracking

    def get_source_info(self, source_id: str) -> Optional[SourceTracking]:
        return self.source_tracking.get(source_id)

    # ---------------------------
    # Timestamp Tracking
    # ---------------------------

    def update_timestamps(self, chunk_id: str, **overrides) -> TimestampTracking:
        ts = TimestampTracking(
            created_date=overrides.get("created_date", now_utc()),
            last_updated=overrides.get("last_updated", now_utc()),
            indexed_date=overrides.get("indexed_date", now_utc()),
            chunk_created=overrides.get("chunk_created", now_utc()),
            last_accessed=overrides.get("last_accessed"),
            next_review=overrides.get("next_review"),
            expiry_date=overrides.get("expiry_date"),
        )
        self.timestamp_tracking[chunk_id] = ts.to_dict()
        self._save()
        return ts

    def get_timestamp_info(self, chunk_id: str) -> Optional[Dict[str, str]]:
        return self.timestamp_tracking.get(chunk_id)

    # ---------------------------
    # Metadata Ops
    # ---------------------------

    def validate_chunk_metadata(self, source_type: str, metadata: Dict[str, Any]):
        return self.field_manager.validate_metadata(source_type, metadata)

    def enhance_metadata(self, metadata: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        enriched = dict(metadata)

        chunk_id = metadata.get("chunk_id")
        source_id = metadata.get("source_id")

        if chunk_id and chunk_id not in self.timestamp_tracking:
            self.update_timestamps(chunk_id)

        if chunk_id:
            enriched.update(self.timestamp_tracking.get(chunk_id, {}))

        if source_id:
            src = self.source_tracking.get(source_id)
            if src:
                enriched.update({
                    "source_system": src.source_system,
                    "source_url": src.source_url,
                    "source_path": src.source_path,
                    "source_version": src.source_version,
                    "ingestion_batch": src.ingestion_batch,
                    "processing_status": src.processing_status,
                })

        enriched.update(self._quality_metrics(enriched))
        enriched.update(self._access_defaults(enriched, source_type))
        return enriched

    # ---------------------------
    # Derived Signals
    # ---------------------------

    def _quality_metrics(self, md: Dict[str, Any]) -> Dict[str, Any]:
        content = md.get("content", "")
        if not content:
            return {"quality_score": 0.0, "freshness_score": 0.0}

        words = content.split()
        avg_word = sum(len(w) for w in words) / max(len(words), 1)

        score = 0.0
        if 50 <= len(content) <= 2000:
            score += 0.3
        if any(x in content for x in ["#", "1.", "-", "*"]):
            score += 0.3
        if 3 <= avg_word <= 8:
            score += 0.2
        if md.get("title"):
            score += 0.2

        last_updated = iso_to_dt(md.get("last_updated"))
        freshness = 0.0
        if last_updated:
            freshness = max(0.0, 1 - (now_utc() - last_updated).days / 365)

        return {
            "quality_score": round(min(score, 1.0), 3),
            "freshness_score": round(freshness, 3),
        }

    def _access_defaults(self, md: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        return {
            "access_level": md.get("access_level", "internal"),
            "allowed_roles": md.get(
                "allowed_roles",
                {
                    "documentation": ["admin", "engineer", "support"],
                    "tickets": ["admin", "support"],
                    "runbooks": ["admin", "engineer"],
                }.get(source_type, []),
            ),
            "usage_count": md.get("usage_count", 0),
            "feedback_score": md.get("feedback_score", 0.0),
        }

# =====================================================
# Singleton
# =====================================================

_tracker: Optional[MetadataTracker] = None

def get_metadata_tracker() -> MetadataTracker:
    global _tracker
    if _tracker is None:
        _tracker = MetadataTracker()
    return _tracker