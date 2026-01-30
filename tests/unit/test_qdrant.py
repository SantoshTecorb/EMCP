from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

COLLECTION = "mcp_tickets"
MODEL = "BAAI/bge-small-en-v1.5"
QUERY = "DAtabase connection issues "

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)

def embed(text: str):
    return model.encode(text, normalize_embeddings=True).tolist()

qdrant = QdrantClient(host="localhost", port=6333)

results = qdrant.query_points(
    collection_name=COLLECTION,
    query=embed(QUERY),
    limit=20
).points

print(f"\nQuery: {QUERY}")
print("=" * 80)

for i, r in enumerate(results, 1):
    payload = r.payload or {}

    print(f"\nResult #{i}")
    print(f"Score    : {r.score:.4f}")
    print(f"Title    : {payload.get('title')}")
    print(f"Chunk ID : {payload.get('chunk_id')}")
    print("Preview:")
    print(payload.get("content", "")[:300])
