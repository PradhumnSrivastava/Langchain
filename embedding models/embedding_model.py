from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Sample texts
documents = [
    "I love studying astronomy and space.",
    "Machine learning is a fascinating field.",
    "The Bhagavad Gita teaches about karma and dharma.",
    "Black holes are mysterious objects in space.",
    "Data science involves statistics and programming."
]

# Step 3: Convert texts to embeddings
doc_embeddings = model.encode(documents)

# Step 4: User query
query = input("Enter your query: ")

# Step 5: Query embedding
query_embedding = model.encode([query])

# Step 6: Similarity calculate
similarities = cosine_similarity(query_embedding, doc_embeddings)

# Step 7: Best match finding
best_index = similarities.argmax()

print("\nMost similar text:")
print(documents[best_index])

print("\nSimilarity score:", similarities[0][best_index])