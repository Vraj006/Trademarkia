import os
import sys
import subprocess
import json
import pickle
import numpy as np
import faiss
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, confusion_matrix
from collections import Counter
from tqdm import tqdm

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ["faiss-cpu", "sentence-transformers", "umap-learn", "hdbscan", "matplotlib", "seaborn", "scikit-learn"]
for pkg in packages:
    try:
        if pkg == "umap-learn":
            import umap
        else:
            __import__(pkg.split('-')[0])
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

def main():
    plot_dir = "analysis_outputs"
    os.makedirs(plot_dir, exist_ok=True)

    print("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    texts = dataset.data
    targets = dataset.target
    target_names = dataset.target_names

    valid_indices = [i for i, text in enumerate(texts) if len(text.strip()) > 20]
    clean_texts = [texts[i] for i in valid_indices]
    clean_targets = [targets[i] for i in valid_indices]
    print(f"Kept {len(clean_texts)} out of {len(texts)} documents after filtering.")

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding documents...")
    embeddings = model.encode(clean_texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)

    print("Applying UMAP for Dimensionality Reduction...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=15, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    print("\n--- 1. Evaluating Cluster Count via Silhouette Score Sweep ---")
    min_cluster_sizes = [10, 25, 50, 75, 100]
    scores = []
    cluster_counts = []
    boundary_case_counts = []
    models_cache = {}

    for size in min_cluster_sizes:
        print(f"Testing HDBSCAN min_cluster_size={size}...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=10, prediction_data=True, metric='euclidean')
        labels = clusterer.fit_predict(reduced_embeddings)
        
        valid_mask = labels != -1
        if len(set(labels[valid_mask])) > 1:
            score = silhouette_score(reduced_embeddings[valid_mask], labels[valid_mask])
        else:
            score = -1.0
            
        scores.append(score)
        cluster_counts.append(len(set(labels[valid_mask])))
        models_cache[size] = clusterer
        
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        b_count = 0
        for memberships in soft_clusters:
            sorted_probs = np.sort(memberships)[::-1]
            if len(sorted_probs) > 1 and sorted_probs[1] > 0.2:
                b_count += 1
        boundary_case_counts.append(b_count)

    plt.figure(figsize=(10, 6))
    plt.plot(min_cluster_sizes, scores, marker='o', color='b', linewidth=2)
    plt.xlabel('min_cluster_size')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. HDBSCAN min_cluster_size')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'silhouette_sweep.png'))
    plt.close()
    
    best_idx = np.argmax(scores)
    best_size = min_cluster_sizes[best_idx]
    best_clusterer = models_cache[best_size]
    print(f"Optimal min_cluster_size selected: {best_size} yielding {cluster_counts[best_idx]} clusters.")
    
    tunable_exploration = []
    for s, c, b in zip(min_cluster_sizes, cluster_counts, boundary_case_counts):
        tunable_exploration.append({
            "min_cluster_size": s,
            "clusters_discovered": c,
            "boundary_cases": b
        })
    with open(os.path.join(plot_dir, "tunable_decision_exploration.json"), "w") as f:
        json.dump(tunable_exploration, f, indent=4)

    soft_clusters = hdbscan.all_points_membership_vectors(best_clusterer)

    print("\n--- 2. Analyzing Cluster Purity ---")
    final_labels = best_clusterer.labels_
    valid_cluster_ids = set(final_labels) - {-1}
    
    purity_analysis = []
    for cluster_id in valid_cluster_ids:
        cluster_indices = np.where(final_labels == cluster_id)[0]
        true_labels = [clean_targets[i] for i in cluster_indices]
        
        counts = Counter(true_labels)
        total = len(true_labels)
        top_3 = counts.most_common(3)
        
        purity = (top_3[0][1] / total) * 100 if total > 0 else 0
        dominants = [{"category": target_names[label], "percentage": round((count/total)*100, 2)} for label, count in top_3]
        
        sample_idx = cluster_indices[np.argmax(soft_clusters[cluster_indices, cluster_id])]
        
        cluster_info = {
            "hdbscan_cluster_id": int(cluster_id),
            "size": total,
            "purity_percentage": round(purity, 2),
            "top_3_categories": dominants,
            "is_cross_cutting_theme": purity < 60.0,
            "sample_document": clean_texts[sample_idx][:300] + "..."
        }
        purity_analysis.append(cluster_info)

    with open(os.path.join(plot_dir, "cluster_purity_analysis.json"), "w") as f:
        json.dump(purity_analysis, f, indent=4)
        
    print(f"Saved purity analysis for {len(valid_cluster_ids)} clusters.")
    
    print("Generating Cluster-Category Confusion Matrix...")
    plt.figure(figsize=(20, 15))
    valid_mask = final_labels != -1
    cm_valid = confusion_matrix(np.array(clean_targets)[valid_mask], final_labels[valid_mask])
    sns.heatmap(cm_valid, annot=False, cmap='Blues', yticklabels=target_names, xticklabels=list(valid_cluster_ids))
    plt.xlabel('Predicted HDBSCAN Cluster ID')
    plt.ylabel('True Newsgroup Category')
    plt.title('Cluster-Category Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'cluster_confusion_matrix.png'))
    plt.close()

    corpus_data = []
    for i, (text, target) in enumerate(zip(clean_texts, clean_targets)):
        memberships = soft_clusters[i]
        corpus_data.append({
            "id": i,
            "text": text,
            "original_label": target_names[target],
            "hard_label": int(final_labels[i]),
            "soft_distribution": memberships.tolist()
        })

    print("\nBuilding FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    faiss.write_index(index, os.path.join(output_dir, 'vector_index.bin'))
    with open(os.path.join(output_dir, 'clusterer.pkl'), 'wb') as f:
        pickle.dump(best_clusterer, f)
    with open(os.path.join(output_dir, 'umap_model.pkl'), 'wb') as f:
        pickle.dump(umap_model, f)
    with open(os.path.join(output_dir, 'corpus.pkl'), 'wb') as f:
        pickle.dump(corpus_data, f)
        
    print("Done! All required artifacts have been generated and saved.")

if __name__ == "__main__":
    main()
