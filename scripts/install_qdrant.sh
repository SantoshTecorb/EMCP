#!/bin/bash

# Qdrant Installation Script for MCP System
# This script installs Qdrant and sets up the vector database

set -e

echo "🚀 Qdrant Installation & Setup for MCP System"
echo "============================================"

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "✅ Docker found - using Docker installation"
    USE_DOCKER=true
elif command -v brew &> /dev/null; then
    echo "✅ Homebrew found - using Homebrew installation"
    USE_BREW=true
else
    echo "❌ Neither Docker nor Homebrew found"
    echo "Please install Docker or Homebrew first"
    exit 1
fi

# Function to install Qdrant with Docker
install_docker() {
    echo "📦 Installing Qdrant with Docker..."
    
    # Create storage directory
    mkdir -p qdrant_storage
    
    # Stop existing container if running
    if docker ps -q -f name=qdrant | grep -q .; then
        echo "🔄 Stopping existing Qdrant container..."
        docker stop qdrant
        docker rm qdrant
    fi
    
    # Pull and run Qdrant
    echo "⬇️  Pulling Qdrant image..."
    docker pull qdrant/qdrant:latest
    
    echo "🚀 Starting Qdrant container..."
    docker run -d \
        --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant
    
    echo "⏳ Waiting for Qdrant to start..."
    sleep 10
    
    # Verify installation
    if curl -s http://localhost:6333/health > /dev/null; then
        echo "✅ Qdrant is running successfully!"
    else
        echo "❌ Qdrant failed to start"
        echo "Check logs with: docker logs qdrant"
        exit 1
    fi
}

# Function to install Qdrant with Homebrew
install_brew() {
    echo "📦 Installing Qdrant with Homebrew..."
    
    # Install Qdrant
    brew install qdrant
    
    # Start Qdrant service
    brew services start qdrant
    
    echo "⏳ Waiting for Qdrant to start..."
    sleep 5
    
    # Verify installation
    if curl -s http://localhost:6333/health > /dev/null; then
        echo "✅ Qdrant is running successfully!"
    else
        echo "❌ Qdrant failed to start"
        echo "Check logs with: brew services list"
        exit 1
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo "🐍 Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "⬇️  Installing required packages..."
    pip install qdrant-client FlagEmbedding torch numpy pandas python-dotenv
    
    echo "✅ Python dependencies installed!"
}

# Function to setup collections
setup_collections() {
    echo "🗂️  Setting up Qdrant collections..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run the setup script
    if [ -f "scripts/setup_vector_namespaces.py" ]; then
        python scripts/setup_vector_namespaces.py --action setup
    else
        echo "❌ Setup script not found"
        echo "Please ensure scripts/setup_vector_namespaces.py exists"
        exit 1
    fi
}

# Function to verify setup
verify_setup() {
    echo "🔍 Verifying setup..."
    
    # Check Qdrant health
    echo "📊 Checking Qdrant health..."
    curl -s http://localhost:6333/health | python -m json.tool
    
    # List collections
    echo ""
    echo "📋 Listing collections..."
    curl -s http://localhost:6333/collections | python -m json.tool
    
    # Check collection counts
    echo ""
    echo "📈 Checking collection statistics..."
    for collection in mcp_documentation mcp_tickets mcp_runbooks; do
        echo "📁 $collection:"
        curl -s "http://localhost:6333/collections/$collection" | python -c "
import sys, json
try:
    data = json.load(sys.stdin)
    result = data.get('result', {})
    print(f'   Points: {result.get(\"points_count\", 0)}')
    print(f'   Vectors: {result.get(\"vectors_count\", 0)}')
    print(f'   Status: Healthy')
except:
    print('   Status: Error')
"
        echo ""
    done
}

# Function to show next steps
show_next_steps() {
    echo "🎉 Installation Complete!"
    echo "======================"
    echo ""
    echo "📚 Next Steps:"
    echo "1. Test search functionality:"
    echo "   python scripts/setup_vector_namespaces.py --action test --source documentation --query 'test'"
    echo ""
    echo "2. Start MCP servers:"
    echo "   python main.py servers"
    echo ""
    echo "3. Test the system:"
    echo "   python main.py demo"
    echo ""
    echo "4. Run interactive mode:"
    echo "   python main.py interactive"
    echo ""
    echo "📖 Documentation:"
    echo "- Installation guide: docs/installation-indexing-guide.md"
    echo "- Namespace config: docs/vector-namespaces.md"
    echo "- Role permissions: docs/role-permissions.md"
    echo ""
    echo "🔧 Useful Commands:"
    echo "- View Qdrant logs: docker logs qdrant"
    echo "- Stop Qdrant: docker stop qdrant"
    echo "- Restart Qdrant: docker restart qdrant"
    echo "- Check collections: curl http://localhost:6333/collections"
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    # Install Qdrant
    if [ "$USE_DOCKER" = true ]; then
        install_docker
    elif [ "$USE_BREW" = true ]; then
        install_brew
    fi
    
    echo ""
    
    # Install Python dependencies
    install_python_deps
    
    echo ""
    
    # Setup collections
    setup_collections
    
    echo ""
    
    # Verify setup
    verify_setup
    
    echo ""
    
    # Show next steps
    show_next_steps
}

# Handle command line arguments
case "${1:-install}" in
    "install")
        main
        ;;
    "docker")
        install_docker
        ;;
    "brew")
        install_brew
        ;;
    "deps")
        install_python_deps
        ;;
    "setup")
        setup_collections
        ;;
    "verify")
        verify_setup
        ;;
    "cleanup")
        echo "🧹 Cleaning up Qdrant installation..."
        if [ "$USE_DOCKER" = true ]; then
            docker stop qdrant 2>/dev/null || true
            docker rm qdrant 2>/dev/null || true
            rm -rf qdrant_storage
        elif [ "$USE_BREW" = true ]; then
            brew services stop qdrant 2>/dev/null || true
            brew uninstall qdrant 2>/dev/null || true
        fi
        echo "✅ Cleanup complete!"
        ;;
    *)
        echo "Usage: $0 {install|docker|brew|deps|setup|verify|cleanup}"
        echo ""
        echo "Commands:"
        echo "  install  - Complete installation (default)"
        echo "  docker   - Install Qdrant with Docker only"
        echo "  brew     - Install Qdrant with Homebrew only"
        echo "  deps     - Install Python dependencies only"
        echo "  setup    - Setup collections only"
        echo "  verify   - Verify installation only"
        echo "  cleanup  - Remove Qdrant installation"
        exit 1
        ;;
esac
