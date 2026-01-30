import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from agents.knowledge_agent import EnterpriseKnowledgeAgent
from core.models import QueryRequest, UserRole

# Setup logging
logging.basicConfig(level=logging.ERROR)

async def run_verification():
    load_dotenv()
    
    # Initialize the agent
    print("🚀 Initializing Enterprise Knowledge Agent...")
    agent = EnterpriseKnowledgeAgent()
    
    # Test cases
    test_queries = [
        {
            "question": "How do I configure the database for production?",
            "role": UserRole.ENGINEER,
            "description": "General documentation query (Engineer)"
        },
        {
            "question": "Any recurring database connection issues with Acme Corp?",
            "role": UserRole.SUPPORT,
            "description": "Ticket-specific query (Support)"
        },
        {
            "question": "Summarize the database maintenance procedure and any related recent incidents.",
            "role": UserRole.ADMIN,
            "description": "Cross-domain query (Admin: Runbooks + Tickets)"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST #{i}: {test['description']}")
        print(f"QUESTION: {test['question']}")
        print(f"ROLE    : {test['role'].value}")
        print(f"{'='*80}")
        
        request = QueryRequest(
            question=test['question'],
            user_role=test['role'],
            max_results=5
        )
        
        try:
            response = await agent.query(request)
            
            print("\n🤖 AGENT ANSWER:")
            print(response.answer)
            
            print("\n📊 METRICS:")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Sources   : {', '.join(response.sources_used)}")
            print(f"Latency   : {response.processing_time:.2f}s")
            
            print("\n💰 ROI METRICS:")
            print(f"Total Tokens: {response.usage.total_tokens}")
            print(f"  - Prompt: {response.usage.prompt_tokens}")
            print(f"  - Completion: {response.usage.completion_tokens}")
            print(f"Est. Cost: ${response.usage.estimated_cost_usd:.6f}")
            
            if response.citations:
                print("\n📖 CITATIONS:")
                for cit in response.citations[:2]:
                    print(f"- [{cit.source_name}] {cit.content_snippet[:100]}...")
                    
        except Exception as e:
            print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(run_verification())
