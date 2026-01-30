import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.prompts import PromptManager
from core.models import UserRole

async def test_prompts():
    manager = PromptManager(prompt_file="data/store/prompts.json")
    
    print("--- Intent Parsing Prompt ---")
    print(manager.get_prompt("intent_parsing")[:100] + "...")
    
    print("\n--- Answer Generation: Engineer ---")
    print(manager.get_prompt("answer_generation", UserRole.ENGINEER))
    
    print("\n--- Answer Generation: Manager ---")
    print(manager.get_prompt("answer_generation", UserRole.MANAGER))
    
    print("\n--- Answer Generation: Default (Sales) ---")
    print(manager.get_prompt("answer_generation", UserRole.SALES))

if __name__ == "__main__":
    asyncio.run(test_prompts())
