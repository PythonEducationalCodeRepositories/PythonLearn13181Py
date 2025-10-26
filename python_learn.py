import numpy as np
from collections import Counter
from db_config import DatabaseConfig
from faiss_indexing import FAISSIndexer
from genai_categorization import ai_categorization


print("="*60)
print("🚀 INCIDENT CATEGORIZATION SYSTEM")
print("Using ChromaDB Embeddings + FAISS Vector Search + Frequency Voting")
print("="*60)


def initialize_system():
    """Initialize database, embeddings, and FAISS index"""
    
    db_config = DatabaseConfig(csv_file="incidents.csv")
    df, category_col = db_config.load_csv()
    embedding_function = db_config.initialize_embedding_function()
    embeddings, descriptions = db_config.generate_embeddings()
    faiss_indexer = FAISSIndexer(similarity_threshold=0.96, top_k=10)
    faiss_indexer.create_index(embeddings)
    db_config.setup_chromadb_collection(descriptions)
    
    return db_config, faiss_indexer


def process_new_incident(new_description, db_config, faiss_indexer):
    """Process new incident using frequency-based category voting"""
    
    print("
" + "="*60)
    print("🔍 PROCESSING NEW INCIDENT")
    print("="*60)
    print(f"Description: {new_description}")
    
    df, category_col = db_config.get_data()
    embedding_function = db_config.embedding_function
    
    print("
[VECTOR] Generating embedding for new incident...")
    try:
        new_embedding = embedding_function([new_description])[0]
        print("✅ Embedding generated")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None
    
    print(f"
[SEARCH] Finding top {faiss_indexer.get_top_k()} similar incidents using FAISS...")
    scores, indices = faiss_indexer.search_similar(new_embedding)
    
    if scores is None or indices is None:
        print("❌ Search failed")
        return None
    
    print(f"✅ Found {len(indices)} similar incidents")
    print("
📊 TOP 10 SIMILAR INCIDENTS:")
    print("-" * 60)
    
    similar_incidents = []
    category_scores = {}
    
    for i, (idx, score) in enumerate(zip(indices, scores)):
        similarity_percent = score * 100
        incident = df.iloc[idx]
        category_tag = str(incident[category_col])
        
        similar_incidents.append({
            'rank': i + 1,
            'index': int(idx),
            'similarity': float(similarity_percent),
            'description': str(incident['Description']),
            'tag': category_tag
        })
        
        if category_tag not in category_scores:
            category_scores[category_tag] = []
        category_scores[category_tag].append(similarity_percent)
        
        print(f"{i+1}. Similarity: {similarity_percent:.2f}%")
        print(f"   Tag: {category_tag}")
        print(f"   Description: {incident['Description'][:70]}...")
        print()
    
    print("
📊 FREQUENCY ANALYSIS:")
    print("-" * 60)
    
    category_list = [inc['tag'] for inc in similar_incidents]
    category_frequency = Counter(category_list)
    
    print(f"Total incidents analyzed: {len(similar_incidents)}")
    print(f"
Category Frequency Distribution:")
    for category, freq in category_frequency.most_common():
        avg_sim = np.mean(category_scores[category])
        print(f"  • {category}: {freq}/{len(similar_incidents)} occurrences (Avg Similarity: {avg_sim:.2f}%)")
    
    most_frequent_category = category_frequency.most_common(1)[0][0]
    most_frequent_count = category_frequency.most_common(1)[0][1]
    
    print(f"
🏆 Most Frequent Category: '{most_frequent_category}' ({most_frequent_count}/{len(similar_incidents)} occurrences)")
    
    avg_similarity_of_most_frequent = np.mean(category_scores[most_frequent_category])
    
    print(f"📈 Average Similarity for '{most_frequent_category}': {avg_similarity_of_most_frequent:.2f}%")
    print(f"🎯 Threshold: {faiss_indexer.get_threshold() * 100}%")
    
    print("
[DECISION] Making categorization decision...")
    
    if avg_similarity_of_most_frequent >= (faiss_indexer.get_threshold() * 100):
        assigned_tag = most_frequent_category
        
        print(f"✅ Average similarity ({avg_similarity_of_most_frequent:.2f}%) >= Threshold ({faiss_indexer.get_threshold() * 100}%)")
        print(f"📌 Assigning tag: '{assigned_tag}'")
        print(f"   (Most frequent category with {most_frequent_count} occurrences)")
        method = "Frequency-Based Vector Search"
        
    else:
        print(f"⚠️ Average similarity ({avg_similarity_of_most_frequent:.2f}%) < Threshold ({faiss_indexer.get_threshold() * 100}%)")
        print("🤖 Calling AI categorization function...")
        
        assigned_tag = ai_categorization(new_description)
        print(f"📌 Assigned tag from AI: '{assigned_tag}'")
        method = "AI Generated"
    
    print("
" + "="*60)
    print("✨ FINAL RESULT")
    print("="*60)
    print(f"New Incident: {new_description}")
    print(f"Assigned Tag: {assigned_tag}")
    print(f"Most Frequent Category: {most_frequent_category} ({most_frequent_count}/{len(similar_incidents)})")
    print(f"Average Similarity (Most Frequent): {avg_similarity_of_most_frequent:.2f}%")
    print(f"Method: {method}")
    print("="*60)
    
    return {
        'description': new_description,
        'assigned_tag': assigned_tag,
        'most_frequent_category': most_frequent_category,
        'frequency_count': most_frequent_count,
        'avg_similarity': avg_similarity_of_most_frequent,
        'similar_incidents': similar_incidents,
        'category_frequency': dict(category_frequency),
        'method': method
    }


if __name__ == "__main__":
    db_config, faiss_indexer = initialize_system()
    
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
        
        result = process_new_incident(user_input, db_config, faiss_indexer)
        
        print("
" + "-"*60)
        print("Enter another incident description (or 'quit' to exit):")


















import numpy as np
from collections import Counter
from db_config import DatabaseConfig
from faiss_indexing import FAISSIndexer
from genai_categorization import ai_categorization


print("="*60)
print("🚀 INCIDENT CATEGORIZATION SYSTEM")
print("Using Weighted Similarity Voting Method")
print("="*60)


def initialize_system():
    """Initialize database, embeddings, and FAISS index"""
    
    db_config = DatabaseConfig(csv_file="incidents.csv")
    df, category_col = db_config.load_csv()
    embedding_function = db_config.initialize_embedding_function()
    embeddings, descriptions = db_config.generate_embeddings()
    faiss_indexer = FAISSIndexer(similarity_threshold=0.96, top_k=10)
    faiss_indexer.create_index(embeddings)
    db_config.setup_chromadb_collection(descriptions)
    
    return db_config, faiss_indexer


def process_new_incident(new_description, db_config, faiss_indexer):
    """Process new incident using weighted similarity voting"""
    
    print("
" + "="*60)
    print("🔍 PROCESSING NEW INCIDENT")
    print("="*60)
    print(f"Description: {new_description}")
    
    df, category_col = db_config.get_data()
    embedding_function = db_config.embedding_function
    
    print("
[VECTOR] Generating embedding for new incident...")
    try:
        new_embedding = embedding_function([new_description])[0]
        print("✅ Embedding generated")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None
    
    print(f"
[SEARCH] Finding top {faiss_indexer.get_top_k()} similar incidents using FAISS...")
    scores, indices = faiss_indexer.search_similar(new_embedding)
    
    if scores is None or indices is None:
        print("❌ Search failed")
        return None
    
    print(f"✅ Found {len(indices)} similar incidents")
    print("
📊 TOP 10 SIMILAR INCIDENTS:")
    print("-" * 60)
    
    similar_incidents = []
    category_weighted_scores = {}
    
    for i, (idx, score) in enumerate(zip(indices, scores)):
        similarity_percent = score * 100
        incident = df.iloc[idx]
        category_tag = str(incident[category_col])
        
        similar_incidents.append({
            'rank': i + 1,
            'index': int(idx),
            'similarity': float(similarity_percent),
            'description': str(incident['Description']),
            'tag': category_tag
        })
        
        # Weighted scoring: weight = similarity^2 (emphasizes high similarity)
        weight = score ** 2
        
        if category_tag not in category_weighted_scores:
            category_weighted_scores[category_tag] = {
                'total_weight': 0,
                'similarities': [],
                'count': 0,
                'max_similarity': 0
            }
        
        category_weighted_scores[category_tag]['total_weight'] += weight
        category_weighted_scores[category_tag]['similarities'].append(similarity_percent)
        category_weighted_scores[category_tag]['count'] += 1
        category_weighted_scores[category_tag]['max_similarity'] = max(
            category_weighted_scores[category_tag]['max_similarity'], 
            similarity_percent
        )
        
        print(f"{i+1}. Similarity: {similarity_percent:.2f}%")
        print(f"   Weight: {weight:.4f}")
        print(f"   Tag: {category_tag}")
        print(f"   Description: {incident['Description'][:70]}...")
        print()
    
    print("
📊 WEIGHTED SIMILARITY ANALYSIS:")
    print("-" * 60)
    
    # Calculate weighted averages and confidence scores
    for category in category_weighted_scores:
        stats = category_weighted_scores[category]
        
        # Weighted average similarity
        stats['weighted_avg_similarity'] = (stats['total_weight'] / stats['count']) * 100
        
        # Simple average similarity
        stats['simple_avg_similarity'] = np.mean(stats['similarities'])
        
        # Confidence score: 70% weighted avg + 30% max similarity
        stats['confidence'] = (stats['weighted_avg_similarity'] * 0.7) + (stats['max_similarity'] * 0.3)
    
    print(f"Total incidents analyzed: {len(similar_incidents)}")
    print(f"
Weighted Category Analysis:
")
    
    # Sort by confidence score
    sorted_categories = sorted(
        category_weighted_scores.items(), 
        key=lambda x: x[1]['confidence'], 
        reverse=True
    )
    
    for category, stats in sorted_categories:
        print(f"  📌 {category}:")
        print(f"     • Occurrences: {stats['count']}/{len(similar_incidents)}")
        print(f"     • Total Weight: {stats['total_weight']:.4f}")
        print(f"     • Weighted Avg Similarity: {stats['weighted_avg_similarity']:.2f}%")
        print(f"     • Simple Avg Similarity: {stats['simple_avg_similarity']:.2f}%")
        print(f"     • Max Similarity: {stats['max_similarity']:.2f}%")
        print(f"     • Confidence Score: {stats['confidence']:.2f}%")
        print()
    
    # Select best category
    best_category = sorted_categories[0][0]
    best_stats = sorted_categories[0][1]
    
    print(f"🏆 Best Category: '{best_category}'")
    print(f"   Confidence Score: {best_stats['confidence']:.2f}%")
    print(f"   Weighted Avg Similarity: {best_stats['weighted_avg_similarity']:.2f}%")
    print(f"🎯 Threshold: {faiss_indexer.get_threshold() * 100}%")
    
    print("
[DECISION] Making categorization decision...")
    
    # Use weighted average similarity for threshold comparison
    if best_stats['weighted_avg_similarity'] >= (faiss_indexer.get_threshold() * 100):
        assigned_tag = best_category
        
        print(f"✅ Weighted avg similarity ({best_stats['weighted_avg_similarity']:.2f}%) >= Threshold ({faiss_indexer.get_threshold() * 100}%)")
        print(f"📌 Assigning tag: '{assigned_tag}'")
        print(f"   (Highest confidence category with {best_stats['count']} occurrences)")
        method = "Weighted Similarity Voting"
        
    else:
        print(f"⚠️ Weighted avg similarity ({best_stats['weighted_avg_similarity']:.2f}%) < Threshold ({faiss_indexer.get_threshold() * 100}%)")
        print("🤖 Calling AI categorization function...")
        
        assigned_tag = ai_categorization(new_description)
        print(f"📌 Assigned tag from AI: '{assigned_tag}'")
        method = "AI Generated"
    
    print("
" + "="*60)
    print("✨ FINAL RESULT")
    print("="*60)
    print(f"New Incident: {new_description}")
    print(f"Assigned Tag: {assigned_tag}")
    print(f"Best Category: {best_category}")
    print(f"Confidence Score: {best_stats['confidence']:.2f}%")
    print(f"Weighted Avg Similarity: {best_stats['weighted_avg_similarity']:.2f}%")
    print(f"Method: {method}")
    print("="*60)
    
    return {
        'description': new_description,
        'assigned_tag': assigned_tag,
        'best_category': best_category,
        'confidence': best_stats['confidence'],
        'weighted_avg_similarity': best_stats['weighted_avg_similarity'],
        'category_stats': category_weighted_scores,
        'similar_incidents': similar_incidents,
        'method': method
    }


if __name__ == "__main__":
    db_config, faiss_indexer = initialize_system()
    
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
        
        result = process_new_incident(user_input, db_config, faiss_indexer)
        
        print("
" + "-"*60)
        print("Enter another incident description (or 'quit' to exit):")