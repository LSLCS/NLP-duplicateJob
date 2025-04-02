import time
import pandas as pd
from vector_search import VectorSearch
import numpy as np
start_time = time.time()
# import torch

# Load dataset
df = pd.read_csv("data/jobs.csv")
df = df.dropna(subset=["companyName"])
job_descriptions = df["jobDescRaw"].astype(str).tolist()

# Get saved embeddings
embeddings = np.load("largeFile/job_embeddings.npy")
print("embeddings' shape", embeddings.shape)

# Get saved FAISS index
dimension = embeddings.shape[1]
vector_search = VectorSearch(dimension, "largeFile/job_index.faiss")

# Considering the top job titles numbers and jobs with duplicated job description numbers,
# using 3000 as the number of jobs for similarity score computation
noCompare = 3000

# Perform vector search once for all embeddings (returns distances and indices)
distances, indices = vector_search.search(embeddings, k=noCompare)

# Convert L2 distance to similarity score in one step
similarity_scores = 1 / (1 + distances[:, 1:])  # Exclude self-match (first column)

# Create DataFrame directly from NumPy arrays
df_similarity = pd.DataFrame({
    "job_id_1": np.repeat(indices[:, 0], noCompare - 1),  # Repeat first column as job1_id
    "job1 Title": np.repeat(df.iloc[indices[:, 0]]["jobTitle"].values, noCompare - 1),
    "job1 company": np.repeat(df.iloc[indices[:, 0]]["companyName"].values, noCompare - 1),
    "job_id_2": indices[:, 1:].flatten(),  # Flatten job2_id indices
    "job2 Title": df.iloc[indices[:, 1:].flatten()]["jobTitle"].values,
    "job2 company": df.iloc[indices[:, 1:].flatten()]["companyName"].values,
    "similarity_score": similarity_scores.flatten()
})
print("\nSimilarity calculated, df_similarity", df_similarity.head(10))

# save csv
df_similarity.to_csv("largeFile/similarity.csv", index=False)

end_time = time.time()
print("Time used", (end_time - start_time)/1, "sec")