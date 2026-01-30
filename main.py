#!/usr/bin/env python3
"""
Multi-Source Enterprise Knowledge Agent with Permissioned Context

Main entry point for the enterprise knowledge system.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.knowledge_agent import EnterpriseKnowledgeAgent
from core.models import UserRole, DataSourceType, QueryRequest
from servers.documentation_server import DocumentationMCPServer
from servers.tickets_server import TicketsMCPServer
from servers.runbooks_server import RunbooksMCPServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def start_servers():
    """Start MCP context servers"""
    logger.info("Starting MCP context servers...")
    
    # Start documentation server
    doc_server = DocumentationMCPServer(port=8001)
    doc_task = asyncio.create_task(doc_server.run_async())
    
    # Start tickets server
    tickets_server = TicketsMCPServer(port=8002)
    tickets_task = asyncio.create_task(tickets_server.run_async())
    
    # Start runbooks server
    runbooks_server = RunbooksMCPServer(port=8003)
    runbooks_task = asyncio.create_task(runbooks_server.run_async())
    
    # Give servers time to start
    await asyncio.sleep(2)
    
    return doc_task, tickets_task, runbooks_task


async def run_demo():
    """Run a demonstration of the system"""
    logger.info("Running Enterprise Knowledge Agent Demo...")
    
    # Initialize the agent
    agent = EnterpriseKnowledgeAgent()
    
    # Demo queries for different user roles
    demo_queries = [
        {
            "question": "How do I authenticate with the API?",
            "role": UserRole.ENGINEER,
            "user_id": "engineer-001"
        },
        {
            "question": "What are common database connection issues?",
            "role": UserRole.SUPPORT,
            "user_id": "support-001"
        },
        {
            "question": "How do I deploy applications to production?",
            "role": UserRole.MANAGER,
            "user_id": "manager-001"
        }
    ]
    
    print("\n" + "="*60)
    print("ENTERPRISE KNOWLEDGE AGENT DEMO")
    print("="*60)
    
    for i, query_info in enumerate(demo_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"User Role: {query_info['role'].value}")
        print(f"Question: {query_info['question']}")
        print("-" * 40)
        
        try:
            request = QueryRequest(
                question=query_info['question'],
                user_role=query_info['role']
            )
            
            response = await agent.query(request)
            
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Sources Used: {len(response.sources_used)}")
            print(f"Processing Time: {response.processing_time:.2f}s")
            
            if response.citations:
                print("\nCitations:")
                for j, citation in enumerate(response.citations, 1):
                    print(f"  {j}. {citation.source_name}")
                    print(f"     Confidence: {citation.confidence_score:.2f}")
            
        except Exception as e:
            logger.error(f"Demo query error: {e}")
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


async def interactive_mode():
    """Run in interactive mode with role selection menu"""
    logger.info("Starting interactive mode...")
    
    agent = EnterpriseKnowledgeAgent()
    
    print("\n" + "="*60)
    print("ENTERPRISE KNOWLEDGE AGENT - INTERACTIVE MODE")
    print("="*60)
    
    # Display available roles
    roles = list(UserRole)
    print("\nAvailable Roles:")
    for i, role in enumerate(roles, 1):
        print(f"  {i}. {role.value.title()}")
    
    print("\nType 'q' to quit at any time")
    print("-" * 60)
    
    selected_role = None
    
    # Role selection
    while selected_role is None:
        try:
            role_input = input("\nSelect your role (enter number 1-5): ").strip()
            
            if role_input.lower() == 'q':
                print("\nGoodbye!")
                return
            
            try:
                role_index = int(role_input) - 1
                if 0 <= role_index < len(roles):
                    selected_role = roles[role_index]
                    print(f"\nRole selected: {selected_role.value.title()}")
                    print("-" * 40)
                else:
                    print(f"Please enter a number between 1 and {len(roles)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
    
    # Question loop
    while True:
        try:
            question = input(f"\n[{selected_role.value.title()}] Your question (or 'q' to quit): ").strip()
            
            if question.lower() == 'q':
                break
            
            if not question:
                print("Please enter a question or 'q' to quit")
                continue
            
            print("\nProcessing your question...")
            
            # Process query
            request = QueryRequest(
                question=question,
                user_role=selected_role
            )
            
            response = await agent.query(request)
            
            print("\n" + "="*50)
            print("RESPONSE")
            print("="*50)
            print(f"\n{response.answer}")
            print(f"\nConfidence: {response.confidence_score:.2f}")
            
            if response.fallback_used:
                print("(Note: Fallback response - insufficient data)")
            
            if response.sources_used:
                print(f"Sources accessed: {', '.join(response.sources_used)}")
            
            if response.citations:
                print(f"\nCitations ({len(response.citations)}):")
                for i, citation in enumerate(response.citations, 1):
                    print(f"  {i}. {citation.source_name}")
                    print(f"     Relevance: {citation.confidence_score:.2f}")
            
            print(f"\nProcessing time: {response.processing_time:.2f}s")
            print(f"Tokens used: {response.usage.total_tokens}")
            print("="*50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Interactive mode error: {e}")
            print(f"\nError processing your question: {e}")
            print("Please try again or enter 'q' to quit")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Agent")
    parser.add_argument(
        "mode", 
        choices=["demo", "interactive", "servers"],
        help="Mode to run the system in"
    )
    parser.add_argument(
        "--no-servers",
        action="store_true",
        help="Don't start MCP servers (use existing ones)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "servers":
            # Start only MCP servers
            doc_task, tickets_task, runbooks_task = await start_servers()
            logger.info("MCP servers started. Press Ctrl+C to stop.")
            await asyncio.gather(doc_task, tickets_task, runbooks_task)
        
        elif args.mode == "demo":
            # Start servers and run demo
            if not args.no_servers:
                doc_task, tickets_task, runbooks_task = await start_servers()
                await asyncio.sleep(3)  # Wait for servers to be ready
            
            await run_demo()
            
            if not args.no_servers:
                # Cancel server tasks
                doc_task.cancel()
                tickets_task.cancel()
                runbooks_task.cancel()
        
        elif args.mode == "interactive":
            # Start servers and interactive mode
            if not args.no_servers:
                doc_task, tickets_task, runbooks_task = await start_servers()
                await asyncio.sleep(3)
            
            await interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Main error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
