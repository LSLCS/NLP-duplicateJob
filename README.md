# NLP-duplicateJob


# **Job Posting Deduplication using NLP & Vector Search**  

## **Overview**  
This project detects duplicate job postings using **text embeddings** and **vector search**. It processes job descriptions, generates embeddings, and applies **approximate nearest neighbor (ANN) search** to identify similar job listings efficiently.  

## **Tech Stack**  
- **Python** (pandas, numpy, sentence-transformers)  
- **FAISS** (for vector search)  
- **Docker & Docker Compose**  
- **Jupyter Notebook** (for EDA)  

---

## **1. Setup Instructions**  

### **Prerequisites**  
Ensure you have the following installed:  
- Python 3.8+  
- Docker & Docker Compose  
- Git  

### **Clone Repository**  
```bash
git clone https://github.com/your-username/job-deduplication.git
cd job-deduplication
```

### **Install Dependencies**  
Using pip:  
```bash
pip install -r requirements.txt
```

Alternatively, using a virtual environment:  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Run with Docker**  
Build and run the containerized app:  
```bash
docker-compose up --build
```

---

## **2. Data Exploration**  
We analyze the job postings dataset (`jobs.csv`), checking:  
✅ Missing values  
✅ Text length distribution  
✅ Existing duplicates  

🔹 **Key Observations:**  
- Many job descriptions contain slight variations but describe the same role.  
- Some postings have redundant listings with identical job titles and companies.  

Full analysis is in [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).  

---

## **3. Embeddings Generation**  
We use **Sentence-Transformers (SBERT)** to convert job descriptions into embeddings.  
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(job_descriptions, show_progress_bar=True)
```
🔹 **Why SBERT?**  
- Efficient & high-quality text representations  
- Works well for semantic similarity tasks  

---

## **4. Vector Search Implementation**  
We use **FAISS** for efficient similarity search.  
```python
import faiss

# Create FAISS index
index = faiss.IndexFlatL2(384)  # 384 is the embedding dimension
index.add(embeddings)
```

### **Search for Similar Jobs**  
```python
D, I = index.search(query_embedding, k=5)  # Find top 5 similar jobs
```

---

## **5. Evaluation**  
To determine duplicate postings, we analyze **cosine similarity distribution** and set a threshold:  
- If **similarity > 0.85**, jobs are considered duplicates.  
- Threshold chosen based on empirical testing & histogram analysis.  

📊 **Results:**  
| Job ID 1 | Job ID 2 | Similarity Score |  
|----------|---------|-----------------|  
| 101      | 205     | 0.92            |  
| 318      | 527     | 0.87            |  

---

## **6. Running the Application**  

### **Step 1: Prepare Data & Generate Embeddings**  
```bash
python src/embeddings.py
```

### **Step 2: Build & Run Vector Search**  
```bash
python src/vector_search.py
```

### **Step 3: Evaluate Results**  
```bash
python src/evaluation.py
```

### **Step 4: Run in Docker**  
```bash
docker-compose up --build
```

---

## **7. File Structure**  
```
job-deduplication/  
│── data/                  # Dataset placeholder (add to .gitignore)  
│── notebooks/             # Jupyter Notebook for EDA  
│── src/                   # Source code  
│   │── embeddings.py      # Embedding generation  
│   │── vector_search.py   # FAISS search implementation  
│   │── evaluation.py      # Similarity threshold & results  
│   │── main.py            # Entry point  
│── docker/  
│   │── Dockerfile         # Container setup  
│   │── docker-compose.yml # Dependency management  
│── requirements.txt       # Dependencies  
│── .env.example          # Placeholder for env variables  
│── README.md              # Documentation  
│── demo.mp4               # 2-minute demo video  
```

---

## **8. Future Enhancements** 🚀  
🔹 Deploy as a **FastAPI** or **Flask API**  
🔹 Use **Milvus** or **Pinecone** for scalable vector search  
🔹 Store embeddings in a database (PostgreSQL, MongoDB)  

---

## **9. Video Demo** 🎥  
📌 [Link to video demo](#) (or include demo.mp4 in submission)  

---

Let me know if you need any changes! 🚀

threshold analysis improvement: I was planning to analyze the similarity of the job titles with the same job description got from EDA. But didn't have time
so just used histogram 