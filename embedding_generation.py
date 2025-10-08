# Install: pip install chromadb pymilvus pandas

import chromadb
from chromadb.utils import embedding_functions
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd

# Load CSV with incidents
df = pd.read_csv('incidents.csv')
print("Loaded Incidents:")
print(df.head())
print()

# Initialize ChromaDB default embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Generate embeddings from DESCRIPTION ONLY (tags stored separately as exact strings)
print("Generating embeddings from DESCRIPTION field only...")
descriptions = df['Description'].tolist()
embeddings = default_ef(descriptions)
print(f"Created {len(embeddings)} embeddings with {len(embeddings[0])} dimensions\n")

# Connect to Milvus
connections.connect(host="localhost", port="19530")

collection_name = "incident_search"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

# Define schema: Embeddings + Exact Scalar Fields
fields = [
    FieldSchema(name="incident_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=200),  # EXACT TAGS
    FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
]

schema = CollectionSchema(fields=fields, description="Incident vector search with exact tags")
collection = Collection(name=collection_name, schema=schema)

# Insert: Embeddings from description + Exact tags as metadata
entities = [
    df['Incident ID'].tolist(),
    df['Description'].tolist(),
    df['Tags'].tolist(),        # Stored as exact string (not embedded)
    df['Date'].tolist(),
    embeddings                   # Vector embeddings from descriptions
]

collection.insert(entities)
print(f"Inserted {len(df)} incidents into Milvus")

# Create COSINE similarity index
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
print("Created index with COSINE similarity\n")

# Search function
def search_incidents(query, top_k=10):
    """Search incidents by description, return exact tags"""
    query_embedding = default_ef([query])

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["incident_id", "description", "tags", "date"]
    )

    return results

# Interactive search
print("="*90)
print("INCIDENT VECTOR SEARCH - Embeddings from Description + Exact Tags Returned")
print("="*90)

while True:
    query = input("\nEnter search query (or 'quit'): ")

    if query.lower() == 'quit':
        break

    if not query.strip():
        continue

    results = search_incidents(query, top_k=10)

    print(f"\n{'Rank':<6} {'ID':<6} {'Similarity':<12} {'Description':<30} {'Exact Tags':<25}")
    print("-" * 90)

    for hits in results:
        for rank, hit in enumerate(hits, 1):
            similarity = hit.distance
            inc_id = hit.entity.get('incident_id')
            desc = hit.entity.get('description')[:28]  # Truncate for display
            tags = hit.entity.get('tags')  # EXACT ORIGINAL TAGS

            print(f"{rank:<6} {inc_id:<6} {similarity:<12.4f} {desc:<30} {tags:<25}")

connections.disconnect("default")