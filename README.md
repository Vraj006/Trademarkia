# Trademarkia Semantic Search API

A highly optimized semantic search engine designed specifically for the 20 Newsgroups dataset. It utilizes non-linear manifold projection, true density-based fuzzy clustering, and a custom multi-table LSH (Locality Sensitive Hashing) semantic cache built from first principles (No Redis).

## Features

1. **Lightweight Embeddings**: Vectors are encoded using the highly efficient `sentence-transformers/all-MiniLM-L6-v2`.
2. **Dense Manifold Discovery (UMAP & HDBSCAN)**: Instead of assuming rigid geometric spheres with PCA/GMM, embeddings are mapped to a non-linear dense plane via UMAP. HDBSCAN natively discovers cluster boundaries, detects outliers/noise, and projects true non-Gaussian fuzzy probability matrices for every document, mathematically proving real-world "boundary cases" spanning overlapping topics.
3. **Rigorous Clustering Analysis**: Automatically sweeps `min_cluster_size` parameters against Silhouette Scores for optimal mathematically-proven density discovery (ignoring random guesswork). Maps purity metrics specifically to detect "cross-cutting themes".
4. **Locality Sensitive Hashing (LSH) Cache**: Pure Python, $O(1)$ constant-time lookup framework utilizing an 8-table XOR projection architecture. 
5. **Cluster-Weighted Cache Matching**: Combines Cosine Similarity (70%) of the local embedding with the normalized Dot Product (30%) of the item's underlying HDBSCAN probability schema. Structurally eliminates exact-phrase false positives (e.g. "Java" Island vs "Java" Data structure) by isolating differing sub-contexts dynamically.

---

## Running Locally

### Step 1. Generate Models (Colab or Local)
You must initialize the Vector DB and Clustering graphs before running the backend.
```bash
# If running locally
pip install -r requirements.txt
python colab_train.py
```
*Note: This generates explicit purity JSON readouts and Silhouette charts in the `/analysis_outputs` folder!*

### Step 2. Start the API
```bash
uvicorn main:app
```
**Access the local interactive Swagger UI** at: `http://127.0.0.1:8000/docs`

---

## Running via Docker

The Docker configuration assumes you have already run `Step 1` above to generate the artifacts in your local `models/` folder. This skips building 200MB+ models into the baseline image layer, significantly speeding up container deployment.

```bash
docker-compose up --build
```
Your cache and API routes are live on `127.0.0.1:8000` via Volume mapping!

