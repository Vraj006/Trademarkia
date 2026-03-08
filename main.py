import os
import pickle
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from cache import LSHSemanticCache
import hdbscan

app_state = {
    "model": None,
    "umap": None,
    "hdbscan": None,
    "index": None,
    "corpus": None,
    "cache": None
}

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: list[dict]
    dominant_cluster: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and vector DB...")
    
    if not os.path.exists("models/vector_index.bin"):
        print("WARNING: 'models/' directory missing. Please run 'colab_train.py' first.")
    else:
        app_state["model"] = SentenceTransformer('all-MiniLM-L6-v2')
        app_state["index"] = faiss.read_index("models/vector_index.bin")
        
        with open("models/clusterer.pkl", "rb") as f:
            app_state["hdbscan"] = pickle.load(f)
            
        with open("models/umap_model.pkl", "rb") as f:
            app_state["umap"] = pickle.load(f)
            
        with open("models/corpus.pkl", "rb") as f:
            app_state["corpus"] = pickle.load(f)
            
        app_state["cache"] = LSHSemanticCache(
            embedding_dim=384, 
            num_hash_bits=8, 
            similarity_threshold=0.75, 
            max_size=1000
        )
    
    yield
    print("Shutting down service...")

app = FastAPI(lifespan=lifespan, title="Trademarkia Semantic Search")

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if app_state["index"] is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    query_emb_raw = app_state["model"].encode([query_text], convert_to_numpy=True)
    query_emb = normalize(query_emb_raw, axis=1)[0]
    
    reduced_emb = app_state["umap"].transform([query_emb])
    cluster_probs = hdbscan.membership_vector(app_state["hdbscan"], reduced_emb)[0]
    
    if np.sum(cluster_probs) == 0:
        top_cluster_id = -1
    else:
        top_cluster_id = int(np.argmax(cluster_probs))
    
    cached_result, matched_query, sim_score = app_state["cache"].get(query_emb, cluster_probs)
    
    if cached_result is not None:
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=matched_query,
            similarity_score=sim_score,
            result=cached_result,
            dominant_cluster=top_cluster_id
        )
        
    k = 3
    D, I = app_state["index"].search(np.array([query_emb]), k)
    
    results = []
    for distance, idx in zip(D[0], I[0]):
        doc = app_state["corpus"][idx]
        results.append({
            "id": doc["id"],
            "original_label": doc["original_label"],
            "text_snippet": doc["text"][:300] + "...",
            "match_score": float(distance)
        })
        
    app_state["cache"].put(query_text, query_emb, cluster_probs, results)
    
    return QueryResponse(
        query=query_text,
        cache_hit=False,
        result=results,
        dominant_cluster=top_cluster_id
    )

@app.get("/cache/stats")
async def get_cache_stats():
    if app_state["cache"] is None:
        return {"error": "Cache not initialized"}
    return app_state["cache"].get_stats()

@app.delete("/cache")
async def flush_cache():
    if app_state["cache"] is None:
        return {"error": "Cache not initialized"}
    app_state["cache"].clear()
    return {"message": "Cache flushed successfully"}
