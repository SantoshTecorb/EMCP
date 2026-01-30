
import asyncio
import os
import logging
from servers.documentation_server import DocumentationMCPServer
from servers.tickets_server import TicketsMCPServer
from servers.runbooks_server import RunbooksMCPServer
from agents.models import UserRole

# Minimize noise
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

async def test_all_servers():
    print("\n🚀 Starting Comprehensive MCP System Verification")
    print("=" * 60)
    
    # 1. Initialize Servers
    doc_server = DocumentationMCPServer()
    tic_server = TicketsMCPServer()
    run_server = RunbooksMCPServer()
    
    # 2. Test Documentation Server
    print("\n[Documentation Server Check]")
    res_doc = await doc_server.search_documentation("API authentication", "engineer")
    print(f"✅ Engineer (Auth) Results: {res_doc.get('total_found')}")
    
    res_doc_prod = await doc_server.search_documentation("API", "product")
    print(f"✅ Product (Auth) Results: {res_doc_prod.get('total_found')}")
    
    # 3. Test Tickets Server
    print("\n[Tickets/CRM Server Check]")
    res_tic = await tic_server.search_tickets("database connection", "leadership")
    print(f"✅ Leadership (Auth) Results: {res_tic.get('total_found')}")
    
    res_tic_filt = await tic_server.search_tickets("database", "support", status="open")
    print(f"✅ Support (Auth) Status:'open' Results: {res_tic_filt.get('total_found')}")
    
    # 4. Test Runbooks Server
    print("\n[Runbooks Server Check]")
    res_run = await run_server.search_runbooks("deployment", "devops")
    print(f"✅ Devops (Auth) Results: {res_run.get('total_found')}")
    
    res_run_sys = await run_server.search_runbooks("failover", "engineer", system="database")
    print(f"✅ Engineer (Auth) System:'database' Results: {res_run_sys.get('total_found')}")
    
    # 5. RBAC Cross-check (Negative cases)
    print("\n[RBAC Cross-check]")
    res_neg1 = await tic_server.search_tickets("sensitive data", "engineer")
    print(f"✅ Engineer access to Tickets: {'Unauthorized' if 'error' in res_neg1 else 'Allowed'}")
    
    res_neg2 = await run_server.search_runbooks("security", "support")
    print(f"✅ Support access to Runbooks: {'Unauthorized' if 'error' in res_neg2 else 'Allowed'}")
    
    print("\n" + "=" * 60)
    print("🏁 System Verification Complete")

if __name__ == "__main__":
    asyncio.run(test_all_servers())
