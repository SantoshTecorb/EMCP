import streamlit as st
import asyncio
import time
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dotenv import load_dotenv
load_dotenv()

from agents.knowledge_agent import EnterpriseKnowledgeAgent
from core.models import UserRole, QueryRequest, ChatMessage

# Page Config
st.set_page_config(
    page_title="MCP Enterprise Knowledge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 2rem;
    }
    .source-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        background-color: #1e293b;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper for running async tasks in Streamlit
def run_async(coro):
    return asyncio.run(coro)

# Initialize Agent
def get_agent():
    try:
        return EnterpriseKnowledgeAgent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        raise

# Remove caching to ensure fresh initialization
agent = get_agent()

# Session State for History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:    
    role_options = [r.value.title() for r in UserRole]
    selected_role = st.selectbox(
        "Select User Role",
        options=role_options,
        index=role_options.index("engineer") if "engineer" in role_options else 0,
        help="Permissions will be enforced based on this role."
    )
    
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Server Status")
    
    # Simple Health Check Visualization
    for name, server in agent.server_manager.servers.items():
        status_color = "🟢" if server.is_healthy else "🔴"
        with st.container():
            col1, col2 = st.columns([1, 4])
            col1.write(status_color)
            col2.write(f"**{name.title()}**")
            col2.caption(f"Endpoint: {server.url}")
    
    if st.button("Sync Servers", use_container_width=True):
        run_async(agent.server_manager.health_check_all())
        st.success("Health checks completed!")
        time.sleep(1)
        st.rerun()

# Main UI
st.markdown('<p class="main-header">Enterprise Knowledge Agent</p>', unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show ROI metrics for assistant responses
        if message["role"] == "assistant" and "tokens" in message:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Tokens: {message['tokens']}")
            with col2:
                st.caption(f"Cost: {message['cost']}")
            with col3:
                st.caption(f"Confidence: {message.get('confidence', 'N/A')}")
        
        if "citations" in message and message["citations"]:
            with st.expander("View Sources & Citations"):
                for i, citation in enumerate(message["citations"]):
                    st.write(f"**[{i+1}] {citation['name']}**")
                    st.caption(f"Confidence: {citation['score']}")
                    st.info(citation["snippet"])

# Query Input
if query := st.chat_input("Ask about technical docs, tickets, or runbooks..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Agent Response
    with st.chat_message("assistant"):
        with st.spinner("Reasoning across enterprise sources..."):
            try:
                # Prepare history for the agent
                history = [
                    ChatMessage(role=m["role"], content=m["content"]) 
                    for m in st.session_state.messages[:-1] # Exclude current query
                ]
                
                request = QueryRequest(
                    question=query,
                    user_role=UserRole(selected_role.lower()),
                    history=history
                )
                
                try:
                    response = run_async(agent.query(request))
                    
                    # DEBUGGING: Show full response structure
                    # st.write("### DEBUG - Full Response Object")
                    # st.json({
                    #     "answer": response.answer,
                    #     "confidence_score": response.confidence_score,
                    #     "sources_used": response.sources_used,
                    #     "citations_count": len(response.citations) if response.citations else 0,
                    #     "processing_time": response.processing_time,
                    #     "usage": response.usage.dict() if response.usage else None
                    # })
                    
                    # Check what's actually in the response
                    if not response.answer or response.answer.strip() == "":
                        st.error("Answer is empty!")
                        # Show what was retrieved
                        st.write("### Retrieved chunks from terminal:")
                        st.write("The agent found relevant chunks (see terminal), but answer generation failed.")
                        st.write("Possible issues:")
                        st.write("1. LLM call failed or returned empty")
                        st.write("2. Context window too large")
                        st.write("3. Response filtering too strict")
                    else:
                        st.markdown(response.answer)                        
                    # ROI Metrics Display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tokens", f"{response.usage.total_tokens}")
                    with col2:
                        st.metric("Est. Cost", f"${response.usage.estimated_cost_usd:.6f}")
                    with col3:
                        st.metric("Latency", f"{response.processing_time:.2f}s")
                    with col4:
                        st.metric("Confidence", f"{response.confidence_score * 100:.0f}%")
                
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                # Format citations for state
                citations_data = [
                    {
                        "name": c.source_name,
                        "score": f"{c.confidence_score:.2f}",
                        "snippet": c.content_snippet
                    } for c in response.citations
                ]
                
                # Save assistant response to state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.answer,
                    "citations": citations_data,
                    "confidence": f"{response.confidence_score * 100:.0f}%",
                    "tokens": response.usage.total_tokens,
                    "cost": f"${response.usage.estimated_cost_usd:.6f}"
                })
                
                # Source Tags
                if response.sources_used:
                    unique_sources = sorted(list(set(response.sources_used)))
                    tags_html = "".join([f'<span class="source-tag">{s}</span>' for s in unique_sources])
                    st.markdown(f'<div style="margin-top: 10px; margin-bottom: 20px;">{tags_html}</div>', unsafe_allow_html=True)

                if response.citations:
                    with st.expander("View Supporting Evidence"):
                        for i, citation in enumerate(response.citations):
                            st.write(f"**[{i+1}] {citation.source_name}**")
                            st.caption(f"Relevance Score: {citation.confidence_score:.2f}")
                            st.info(citation.content_snippet)
                
            except Exception as e:
                st.error(f"Internal Error: {str(e)}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
