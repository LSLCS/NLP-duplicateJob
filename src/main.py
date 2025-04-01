import time
import pandas as pd
from vector_search import VectorSearch
import numpy as np
start_time = time.time()
# from sklearn.metrics.pairwise import cosine_similarity
# import torch

# Load dataset
df = pd.read_csv("data/jobs.csv")
df = df.dropna(subset=["companyName"])
job_descriptions = df["jobDescRaw"].astype(str).tolist()

# Get saved embeddings
embeddings = np.load("job_embeddings.npy")
print("embeddings' shape", embeddings.shape)

# Get saved FAISS index
dimension = embeddings.shape[1]
vector_search = VectorSearch(dimension, "job_index.faiss")


# # Function to find job similarity
def get_similarity(query_embedding, top_k=5):
    # query_embedding = model.encode([query_text]).astype("float32")
    # distances, indices = index.search(query_embedding, top_k)
    distances, indices = vector_search.search(query_embedding, k=top_k)
    results = []
    for i in range(top_k):
        job_id = indices[0][i]
        similarity = 1 / (1 + distances[0][i])  # Convert L2 to similarity
        
        results.append((df.iloc[job_id]["jobTitle"], df.iloc[job_id]["companyName"], similarity))
        # print(f"jobTitle: {df.iloc[job_id]['jobTitle']}, Similarity: {distances[0][i]}")
    # print("results: ",results)
    return results

# Example Query
# query = "Software Engineer with Python experience"
# print("embeddings[0] for ",df.iloc[0])
similar_jobs = get_similarity(embeddings[1], 50)

print("\nSimilarity calculated, similar_jobs:", similar_jobs)
for job in similar_jobs[:10]:
    print(f"Title: {job[0]}, Company: {job[1]}, Similarity: {job[2]:.4f}")

end_time = time.time()
print("Time used", (end_time - start_time)/1, "sec")