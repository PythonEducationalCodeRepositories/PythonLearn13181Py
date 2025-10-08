import pandas as pd
import numpy as np
import faiss
import chromadb
from chromadb.utils import embedding_functions
import re
from collections import Counter

# ============================================
# CONFIGURATION
# ============================================
CSV_FILE = "incidents.csv"  # Your CSV file in current directory
SIMILARITY_THRESHOLD = 0.96  # 96% threshold
TOP_K = 10  # Top 10 similar incidents

print("="*60)
print("🚀 INCIDENT CATEGORIZATION SYSTEM")
print("Using ChromaDB Embeddings + FAISS Vector Search")
print("="*60)

# ============================================
# STEP 1: LOAD CSV DATA
# ============================================
print("
[STEP 1] Loading incident data from CSV...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Successfully loaded {len(df)} incidents")
    print(f"Columns: {list(df.columns)}")
    print(f"
First few records:")
    print(df.head())
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# Clean the data
if 'Description' not in df.columns:
    print("❌ 'Description' column not found in CSV")
    exit()

# Find category column
category_col = None
for col in ['Category', 'Tag', 'Type', 'Department', 'category', 'tag']:
    if col in df.columns:
        category_col = col
        break

if category_col:
    print(f"✅ Found category column: '{category_col}'")
else:
    print("⚠️ No category column found. Will use index as category.")
    df['Category'] = df.index
    category_col = 'Category'

# ============================================
# STEP 2: INITIALIZE CHROMADB EMBEDDING
# ============================================
print("
[STEP 2] Initializing ChromaDB Embedding Function...")
try:
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    print("✅ ChromaDB DefaultEmbeddingFunction initialized")
except Exception as e:
    print(f"❌ Error initializing embedding function: {e}")
    exit()

# ============================================
# STEP 3: GENERATE EMBEDDINGS
# ============================================
print("
[STEP 3] Generating embeddings for all incident descriptions...")
try:
    descriptions = df['Description'].fillna("").tolist()
    
    embeddings = embedding_function(descriptions)
    embeddings_array = np.array(embeddings).astype('float32')
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
except Exception as e:
    print(f"❌ Error generating embeddings: {e}")
    exit()

# ============================================
# STEP 4: CREATE FAISS INDEX
# ============================================
print("
[STEP 4] Creating FAISS index for vector search...")
try:
    dimension = embeddings_array.shape[1]
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    print(f"Index type: IndexFlatIP (Cosine Similarity)")
except Exception as e:
    print(f"❌ Error creating FAISS index: {e}")
    exit()

# ============================================
# STEP 5: CHROMADB COLLECTION
# ============================================
print("
[STEP 5] Storing data in ChromaDB collection...")
try:
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection(name="incidents")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="incidents",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [str(i) for i in range(len(df))]
    metadatas = []
    
    for idx, row in df.iterrows():
        metadata = {
            "incident_id": str(row.get('IncidentID', idx)),
            "date": str(row.get('Date', '')),
            "category": str(row[category_col])
        }
        metadatas.append(metadata)
    
    collection.add(
        documents=descriptions,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Stored {len(descriptions)} incidents in ChromaDB")
except Exception as e:
    print(f"⚠️ ChromaDB storage warning: {e}")

# ============================================
# FUNCTION: SIMPLE KEYWORD-BASED CATEGORIZATION
# ============================================
def categorize_with_keywords(description):
    """Simple rule-based categorization using keywords"""
    print("
[GENAI] Generating category using keyword analysis...")
    
    description_lower = description.lower()
    
    # Define keyword categories
    categories = {
        'IT': ['server', 'network', 'computer', 'software', 'database', 'system', 'crash', 'connectivity', 'internet'],
        'Safety': ['fire', 'alarm', 'emergency', 'evacuation', 'injury', 'accident', 'hazard'],
        'Facilities': ['power', 'outage', 'water', 'leakage', 'hvac', 'heating', 'cooling', 'building', 'maintenance'],
        'Hardware': ['equipment', 'device', 'machine', 'hardware', 'printer', 'monitor', 'keyboard'],
        'Security': ['breach', 'unauthorized', 'access', 'theft', 'intrusion', 'lock', 'camera'],
        'HR': ['employee', 'staff', 'personnel', 'leave', 'attendance', 'payroll'],
    }
    
    # Count keyword matches
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            scores[category] = score
    
    # Get category with highest score
    if scores:
        category = max(scores, key=scores.get)
        print(f"✅ Generated category: {category} (matched {scores[category]} keywords)")
        return category
    else:
        # Extract main nouns as category
        words = re.findall(r'\b[a-zA-Z]+\b', description)
        if len(words) > 0:
            # Use most common word
            common_words = Counter(words).most_common(3)
            category = common_words[0][0].capitalize()
            print(f"✅ Generated category: {category} (from text analysis)")
            return category
        else:
            print(f"✅ Generated category: Uncategorized")
            return "Uncategorized"

# ============================================
# MAIN FUNCTION: PROCESS NEW INCIDENT
# ============================================
def process_new_incident(new_description):
    """Process new incident and assign category"""
    print("
" + "="*60)
    print("🔍 PROCESSING NEW INCIDENT")
    print("="*60)
    print(f"Description: {new_description}")
    
    # Generate embedding
    print("
[VECTOR] Generating embedding for new incident...")
    try:
        new_embedding = embedding_function([new_description])
        new_embedding_array = np.array(new_embedding).astype('float32')
        faiss.normalize_L2(new_embedding_array)
        print("✅ Embedding generated")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None
    
    # FAISS search
    print(f"
[SEARCH] Finding top {TOP_K} similar incidents using FAISS...")
    try:
        scores, indices = index.search(new_embedding_array, TOP_K)
        scores = scores[0]
        indices = indices[0]
        
        print(f"✅ Found {len(indices)} similar incidents")
        print("
📊 TOP 10 SIMILAR INCIDENTS:")
        print("-" * 60)
        
        similar_incidents = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            similarity_percent = score * 100
            incident = df.iloc[idx]
            category = incident[category_col]
            
            similar_incidents.append({
                'rank': i + 1,
                'index': idx,
                'similarity': similarity_percent,
                'description': incident['Description'],
                'category': category
            })
            
            print(f"{i+1}. Similarity: {similarity_percent:.2f}%")
            print(f"   Category: {category}")
            print(f"   Description: {incident['Description'][:70]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None
    
    # Calculate average similarity
    avg_similarity = np.mean(scores) * 100
    print(f"
[ANALYSIS] Average Similarity Score: {avg_similarity:.2f}%")
    print(f"Threshold: {SIMILARITY_THRESHOLD * 100}%")
    
    # Decision
    print("
[DECISION] Making categorization decision...")
    
    if avg_similarity >= (SIMILARITY_THRESHOLD * 100):
        # Use existing category
        top_incident = similar_incidents[0]
        assigned_category = top_incident['category']
        print(f"✅ Average similarity ({avg_similarity:.2f}%) >= Threshold ({SIMILARITY_THRESHOLD * 100}%)")
        print(f"📌 Assigning category from MOST similar incident: '{assigned_category}'")
        print(f"   (Top match had {top_incident['similarity']:.2f}% similarity)")
        method = "Existing Tag"
    else:
        # Generate new category
        print(f"⚠️ Average similarity ({avg_similarity:.2f}%) < Threshold ({SIMILARITY_THRESHOLD * 100}%)")
        print("🤖 Generating new category using keyword analysis...")
        assigned_category = categorize_with_keywords(new_description)
        print(f"📌 Assigned NEW category: '{assigned_category}'")
        method = "Generated Category"
    
    # Final result
    print("
" + "="*60)
    print("✨ FINAL RESULT")
    print("="*60)
    print(f"New Incident: {new_description}")
    print(f"Assigned Category: {assigned_category}")
    print(f"Average Similarity: {avg_similarity:.2f}%")
    print(f"Method: {method}")
    print("="*60)
    
    return {
        'description': new_description,
        'category': assigned_category,
        'avg_similarity': avg_similarity,
        'similar_incidents': similar_incidents,
        'method': method
    }

# ============================================
# STEP 6: USER INPUT
# ============================================
print("

" + "="*60)
print("🎯 READY FOR NEW INCIDENT INPUT")
print("="*60)

print("
Enter new incident description (or 'quit' to exit):")
while True:
    user_input = input("
>>> ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("
👋 Exiting system. Goodbye!")
        break
    
    if not user_input:
        print("⚠️ Please enter a valid description")
        continue
    
    # Process the incident
    result = process_new_incident(user_input)
    
    print("
" + "-"*60)
    print("Enter another incident description (or 'quit' to exit):")