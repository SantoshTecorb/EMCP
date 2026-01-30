# Multi-Source Enterprise Knowledge Agent with Permissioned Context

A sophisticated enterprise-grade system that combines MCP (Model Context Protocol) servers for context management with LangChain for retrieval and generation, providing role-based access control and citation-required responses.

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Qdrant vector database
- Groq API key (or alternative LLM provider)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MCP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and configuration

# Start Qdrant (using Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# Index sample data (optional)
python scripts/indexing_pipeline.py

python3 main.py interactive --no-servers

```

### Web Interface

```bash
# Launch the Streamlit web interface
streamlit run app.py
```

Visit `http://localhost:8501` to access the web interface.

## Features

### Core Features
- **Multiple MCP Context Servers**: Separate servers for documentation, tickets, and runbooks
- **Role-Based Access Control**: Fine-grained permissions based on user roles
- **LangChain Integration**: Retrieval-first pipeline with confidence gating
- **Citation-Required Responses**: All answers include source citations
- **Fallback Behavior**: Graceful handling when data is missing or insufficient
- **Real-time Processing**: Sub-second response times with async architecture

### Advanced Features
- **Cross-Context Comparison**: Compare information across different data sources
- **Feedback Loop**: Flag outdated sources and improve quality over time
- **Admin Dashboard**: Monitor system health and source quality
- **Real-time Health Checks**: Continuous monitoring of context servers
- **ROI Tracking**: Token usage and cost monitoring
- **Conversation Memory**: Context-aware conversations
- **Analytics Dashboard**: Usage patterns and performance metrics



## System Architecture Overview

### Core Components

**1. Knowledge Orchestrator**
- Central coordinator that processes user queries through modular pipeline
- Enforces permission checks based on user role
- Coordinates retrieval across multiple sources
- Validates answer quality and citation completeness
- Manages conversation context and history

**2. MCP Context Servers**
Three independent FastAPI servers, each managing a specific knowledge domain:
- **Documentation Server** (port 8001): Company docs, wikis, PDFs
- **Tickets Server** (port 8002): Support tickets, CRM data
- **Runbooks Server** (port 8003): Engineering procedures, deployment guides

**3. Vector Database (Qdrant)**
- Separate collection namespace per context server
- Stores document chunks as searchable embeddings
- Maintains metadata (source, timestamp, ownership)
- Semantic search with cosine similarity

**4. Modular Pipeline**
- **Query Parser**: Extracts intent and entities from natural language
- **Knowledge Retriever**: Fetches relevant chunks from multiple sources
- **Knowledge Generator**: Creates cited responses using LLM
- **Permission Manager**: Enforces role-based access control

### Permission Enforcement

- **Server-Level**: Each MCP server validates allowed roles
- **Agent-Level**: Orchestrator filters available servers before querying
- **Response-Level**: Unauthorized access attempts return clear error messages
- **Audit Trail**: All access attempts logged for security monitoring
- **Zero Trust**: No cross-contamination between permission boundaries


