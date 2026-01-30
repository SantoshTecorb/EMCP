import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from core.models import UserRole

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages system prompts with role-based overrides"""
    
    def __init__(self, prompt_file: str = "data/store/prompts.json"):
        self.prompt_file = Path(prompt_file)
        self.prompts: Dict[str, Dict[str, str]] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from JSON file"""
        try:
            if self.prompt_file.exists():
                with open(self.prompt_file, "r") as f:
                    self.prompts = json.load(f)
                logger.info(f"Loaded prompts from {self.prompt_file}")
            else:
                logger.warning(f"Prompt file {self.prompt_file} not found. Using empty defaults.")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self.prompts = {}

    def get_prompt(self, task_id: str, role: Optional[UserRole] = None) -> str:
        """
        Retrieve a prompt for a specific task and role.
        Falls back to 'default' if role-specific prompt is missing.
        """
        task_prompts = self.prompts.get(task_id, {})
        
        # 1. Try role-specific
        if role:
            role_val = role.value if hasattr(role, 'value') else str(role)
            if role_val in task_prompts:
                return task_prompts[role_val]
        
        # 2. Try default
        return task_prompts.get("default", "No prompt found for this task.")

    def refresh(self):
        """Reload prompts from disk"""
        self._load_prompts()
