from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

def check_qdrant():
    load_dotenv()
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    print("--- Qdrant Collection Status ---")
    collections = client.get_collections().collections
    for c in collections:
        info = client.get_collection(c.name)
        print(f"Collection: {c.name}")
        print(f"  Points count: {info.points_count}")
        print(f"  Dimension: {info.config.params.vectors.size}")
        print(f"  Distance: {info.config.params.vectors.distance}")
        print("-" * 30)

if __name__ == "__main__":
    check_qdrant()
