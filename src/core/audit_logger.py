"""
Audit Logger for MCP System
Persists all AI interactions, token usage, and costs to a JSONL file.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class AuditLogger:
    """Thread-safe JSONL logger for AI interactions"""
    
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.jsonl"
        self.logger = logging.getLogger("mcp_audit")
        
        # Setup specific logger if not already setup
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_interaction(self, question: str, role: str, response_data: Dict[str, Any]):
        """Log a complete user interaction with the system"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": question,
            "user_role": role,
            "answer_length": len(response_data.get("answer", "")),
            "confidence": response_data.get("confidence_score", 0.0),
            "sources": response_data.get("sources_used", []),
            "latency": response_data.get("processing_time", 0.0),
            "token_usage": response_data.get("usage", {}),
            "fallback_used": response_data.get("fallback_used", False)
        }
        
        # Write to JSONL
        self.logger.info(json.dumps(log_entry))
