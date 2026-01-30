"""
MCP Client for managing context servers
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import httpx

from core.models import ContextServer, UserRole, DataSourceType

logger = logging.getLogger(__name__)


class ContextServerManager:
    """Manages connections to MCP context servers"""
    
    def __init__(self):
        self.servers: Dict[str, ContextServer] = {}
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create an active httpx client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=90.0)
        return self._client
    
    def register_server(self, server: ContextServer):
        """Register a new context server"""
        self.servers[server.name] = server
        logger.info(f"Registered server: {server.name} at {server.url}")
    
    def unregister_server(self, server_name: str):
        """Unregister a context server"""
        if server_name in self.servers:
            del self.servers[server_name]
            logger.info(f"Unregistered server: {server_name}")
    
    async def health_check_all(self):
        """Perform health checks on all servers"""
        tasks = []
        for server_name, server in self.servers.items():
            task = self._health_check_server(server_name, server)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _health_check_server(self, server_name: str, server: ContextServer):
        """Check health of a specific server"""
        try:
            client = await self.get_client()
            response = await client.get(f"{server.url}/health")
            if response.status_code == 200:
                server.is_healthy = True
                server.last_check = datetime.now()
                logger.info(f"Server {server_name} is healthy")
            else:
                server.is_healthy = False
                logger.warning(f"Server {server_name} returned status {response.status_code}")
        except Exception as e:
            server.is_healthy = False
            logger.error(f"Health check failed for {server_name}: {e}")
    
    async def search_server(
        self,
        server_name: str,
        query: str,
        max_results: int = 10,
        min_confidence: float = 0.7,
        **kwargs
    ):
        """Search a specific server with optional filters"""
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")
        
        if not server.is_healthy:
            # Silently attempt to connect anyway if health check hasn't run
            pass
        
        try:
            payload = {
                "query": query,
                "max_results": max_results,
                "min_confidence": min_confidence
            }
            payload.update(kwargs)
            
            client = await self.get_client()
            response = await client.post(
                f"{server.url}/search",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Search failed for {server_name}: {e}")
            raise
    
    async def get_server_metadata(self, server_name: str):
        """Get metadata from a specific server"""
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Server {server_name} not found")
        
        try:
            client = await self.get_client()
            response = await client.get(f"{server.url}/metadata")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Metadata fetch failed for {server_name}: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
