import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def evaluate_similarity(df_similar_jobs, threshold=0.8):
    """
    Evaluates the effectiveness of the similarity threshold.
    
    Args:
        df_similar_jobs (pd.DataFrame): DataFrame containing job pairs and similarity scores.
        threshold (float): Similarity threshold for considering jobs as duplicates.

    Returns:
        dict: Evaluation results including precision, recall, and F1-score.
    """
    # Apply threshold to classify pairs as duplicates
    df_similar_jobs["is_duplicate"] = df_similar_jobs["similarity_score"] >= threshold

    # If labeled data is available, compare predicted duplicates with ground truth
    # Example ground truth: A CSV mapping job_id_1 & job_id_2 to actual duplicates
    try:
        df_ground_truth = pd.read_csv("ground_truth.csv")  # Needs manually labeled duplicates
        y_true = df_ground_truth["is_duplicate"].astype(int).values
        y_pred = df_similar_jobs["is_duplicate"].astype(int).values
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    except FileNotFoundError:
        print("⚠️ No ground truth found. Running without precision/recall.")
        precision, recall, f1 = None, None, None

    # Histogram of similarity scores
    plt.hist(df_similar_jobs["similarity_score"], bins=50, color="skyblue", edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="dashed", label=f"Threshold = {threshold}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.show()

    return {"Precision": precision, "Recall": recall, "F1-score": f1}

if __name__ == "__main__":
    # Load detected similar job pairs
    df_similar_jobs = pd.read_csv("similar_jobs.csv")
    evaluation_results = evaluate_similarity(df_similar_jobs, threshold=0.8)
    print("Evaluation Results:", evaluation_results)
