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
# print(type(embeddings))
vector_search = VectorSearch(dimension, "job_index.faiss")
D, I = vector_search.search(embeddings, k=100)
print(type(D), D.shape)
print(type(I), I.shape)