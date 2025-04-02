import pandas as pd
from embedding_util import generate_embeddings
from vector_search import VectorSearch

# ####################################################################
# Load dataset
df = pd.read_csv("data/jobs.csv")
df = df.dropna(subset=["companyName"])
job_descriptions = df["jobDescRaw"].astype(str).tolist()

# ####################################################################
# Generate embeddings
embeddings = generate_embeddings(job_descriptions)
print("embeddings", embeddings.shape)


# ####################################################################
# Initialize FAISS index
dimension = embeddings.shape[1]
vector_search = VectorSearch(dimension)

# ####################################################################
# Add embeddings to the FAISS index
vector_search.add_embeddings(embeddings)
print(f"Indexed {len(embeddings)} job descriptions.")

# ####################################################################
# Save the FAISS index to a file
vector_search.save_index("largeFile/job_index.faiss")

