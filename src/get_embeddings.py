import pandas as pd
from embedding_util import generate_embeddings
from vector_search import VectorSearch

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import torch

# Load dataset
df = pd.read_csv("data/jobs.csv")
df = df.dropna(subset=["companyName"])
job_descriptions = df["jobDescRaw"].astype(str).tolist()

# Generate embeddings
embeddings = generate_embeddings(job_descriptions)
# embeddings = generate_embeddings(job_descriptions)
print("embeddings", embeddings.shape)


# Initialize FAISS index
dimension = embeddings.shape[1]
vector_search = VectorSearch(dimension)

# Add embeddings to the FAISS index
vector_search.add_embeddings(embeddings)
print(f"Indexed {len(embeddings)} job descriptions.")

# Save the FAISS index to a file
# faiss.write_index(index, 'job_embeddings.index') 
vector_search.save_index("largeFile/job_index.faiss")

