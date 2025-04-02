import time
import pandas as pd
from vector_search import VectorSearch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
start_time = time.time()
# import torch

# ####################################################################
# Load dataset
df = pd.read_csv("data/jobs.csv")
df = df.dropna(subset=["companyName"])
job_descriptions = df["jobDescRaw"].astype(str).tolist()

# ####################################################################
# Get saved embeddings
embeddings = np.load("largeFile/job_embeddings.npy")

# ####################################################################
# Get saved FAISS index
dimension = embeddings.shape[1]
vector_search = VectorSearch(dimension, "largeFile/job_index.faiss")

# ####################################################################
# Perform vector search

# Considering the top job titles numbers and jobs with duplicated job description numbers,
# using 3000 as the number of jobs for similarity score computation
noCompare = 3000
# Perform vector search once for all embeddings (returns distances and indices)
distances, indices = vector_search.search(embeddings, k=noCompare)

# Convert L2 distance to similarity score in one step
similarity_scores = 1 / (1 + distances[:, 1:])  # Exclude self-match (first column)

# ####################################################################
# Create similarity score DataFrame 
df_similarity = pd.DataFrame({
    "job_id_1": np.repeat(indices[:, 0], noCompare - 1),  
    "job_id_2": indices[:, 1:].flatten(),  # Flatten job2_id indices
    "similarity_score": similarity_scores.flatten()
})

# ####################################################################
# # Histogram of similarity scores
plt.hist(df_similarity['similarity_score'], bins=50, alpha=0.75, color='blue')
plt.title("Distribution of Similarity Scores")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.savefig("similarity_score_histogram.png")
plt.close()

# ####################################################################
# From the histogram, set threshold to 0.85
threshold = 0.85
df_similarity['is_duplicate'] = df_similarity['similarity_score'] >= threshold
df_duplicates = df_similarity[df_similarity['is_duplicate'] == True]
df_duplicates = df_duplicates[['job_id_1', 'job_id_2', 'similarity_score']]
df_duplicates.to_csv("largeFile/detected_duplicates.csv", index=False)

# ####################################################################
# Show Top Job Titles with Duplicated Job Postings
# Merge job title and company name
unique_job_ids = df_duplicates['job_id_1'].drop_duplicates().reset_index(drop=True)
df_duplicates_info = pd.merge(unique_job_ids, df[['jobTitle', 'companyName']], left_on='job_id_1', right_index=True, suffixes=('_1', '_2'))

# Get the top 10 most frequent job titles
top_job_titles = df_duplicates_info['jobTitle'].value_counts().head(10)

# Create a bar plot
plt.figure(figsize=(25, 6))
sns.barplot(x=top_job_titles.values, y=top_job_titles.index)
plt.title("Top Job Titles with Duplicated Job Postings")
plt.xlabel("Count of Duplicated Job Postings")
plt.ylabel("Job Titles")
plt.savefig("top_job_titles.png")
plt.close()

# ####################################################################
#  Show Top Companies with Duplicated Job Postings

# Get the top 10 most frequent companies
top_companies = df_duplicates_info['companyName'].value_counts().head(10)

# Create a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x=top_companies.values, y=top_companies.index)
plt.title("Top Companies with Duplicated Job Postings")
plt.xlabel("Count of Duplicated Job Postings")
plt.ylabel("Companies")
plt.savefig("top_companies.png")
plt.close()

# ####################################################################
# Output some example of duplicates
print("========The first 20 duplication example===========")
print(df_duplicates.iloc[:20])
print("========The last 20 duplication example========")
print(df_duplicates.iloc[-20:])

# Time used
end_time = time.time()
print("Time used", (end_time - start_time)/1, "sec")