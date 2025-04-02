import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# ####################################################################
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ####################################################################
# Generate embeddings
def generate_embeddings(job_descriptions, save_path="largeFile/job_embeddings.npy", as_tensor=False):
    """
    Generate embeddings for job descriptions and save them.
    
    :param job_descriptions: List of job descriptions
    :param save_path: Path to save the embeddings
    :param as_tensor: Whether to return embeddings as PyTorch tensors
    :return: Generated embeddings
    """
    print("Generating embeddings...")
    embeddings = model.encode(job_descriptions, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    # Save embeddings to a file for later use 
    np.save(save_path, embeddings)
  
    return embeddings