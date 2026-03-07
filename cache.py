import numpy as np
import time

class LSHSemanticCache:
    def __init__(self, embedding_dim=384, num_tables=8, num_hash_bits=12, similarity_threshold=0.87, max_size=1000):
        self.embedding_dim = embedding_dim
        self.num_tables = num_tables
        self.num_hash_bits = num_hash_bits
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        
        np.random.seed(42)
        self.hyperplanes = [np.random.randn(self.embedding_dim, self.num_hash_bits) for _ in range(self.num_tables)]
        
        self.tables = [{} for _ in range(self.num_tables)]
        self.store = {}
        self.next_entry_id = 0
        
        self.hit_count = 0
        self.miss_count = 0
        
        self.alpha_embedding = 0.70
        self.alpha_cluster = 0.30

    def _hash_vector(self, embedding, table_idx):
        projections = np.dot(embedding, self.hyperplanes[table_idx])
        return "".join(['1' if x > 0 else '0' for x in projections])
        
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2)
        
    def _distribution_similarity(self, dist1, dist2):
        d1 = np.array(dist1)
        d2 = np.array(dist2)
        d1_norm = d1 / (np.linalg.norm(d1) + 1e-9)
        d2_norm = d2 / (np.linalg.norm(d2) + 1e-9)
        return np.dot(d1_norm, d2_norm)

    def get(self, query_emb, cluster_distribution):
        candidate_ids = set()
        
        for i in range(self.num_tables):
            bucket_key = self._hash_vector(query_emb, i)
            if bucket_key in self.tables[i]:
                candidate_ids.update(self.tables[i][bucket_key])
                
        if not candidate_ids:
            self.miss_count += 1
            return None, None, None
            
        best_match = None
        best_score = -1
        
        for entry_id in candidate_ids:
            if entry_id not in self.store: 
                continue
            entry = self.store[entry_id]
            
            semantic_sim = self._cosine_similarity(query_emb, entry['embedding'])
            cluster_sim = self._distribution_similarity(cluster_distribution, entry['cluster_distribution'])
            
            final_score = (self.alpha_embedding * semantic_sim) + (self.alpha_cluster * cluster_sim)
            
            if final_score > best_score:
                best_score = final_score
                best_match = entry
                
        if best_match and best_score >= self.similarity_threshold:
            self.hit_count += 1
            best_match['timestamp'] = time.time()
            return best_match['result'], best_match['query_text'], float(best_score)
            
        self.miss_count += 1
        return None, None, None
        
    def put(self, query_text, query_emb, cluster_distribution, result):
        if len(self.store) >= self.max_size:
            self._evict_lru()
            
        entry_id = self.next_entry_id
        self.next_entry_id += 1
        
        entry = {
            "id": entry_id,
            "query_text": query_text,
            "embedding": query_emb,
            "cluster_distribution": cluster_distribution,
            "result": result,
            "timestamp": time.time()
        }
        
        self.store[entry_id] = entry
        
        for i in range(self.num_tables):
            bucket_key = self._hash_vector(query_emb, i)
            if bucket_key not in self.tables[i]:
                self.tables[i][bucket_key] = []
            self.tables[i][bucket_key].append(entry_id)
        
    def _evict_lru(self):
        if not self.store:
            return
            
        oldest_entry_id = min(self.store.keys(), key=lambda k: self.store[k]['timestamp'])
        oldest_entry = self.store[oldest_entry_id]
        
        for i in range(self.num_tables):
            bucket_key = self._hash_vector(oldest_entry['embedding'], i)
            if bucket_key in self.tables[i]:
                try:
                    self.tables[i][bucket_key].remove(oldest_entry_id)
                    if not self.tables[i][bucket_key]:
                        self.tables[i].pop(bucket_key, None)
                except ValueError:
                    pass
                    
        del self.store[oldest_entry_id]

    def get_stats(self):
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        return {
            "total_entries": len(self.store),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "max_capacity": self.max_size,
            "similarity_threshold": self.similarity_threshold,
            "lsh_tables_count": self.num_tables
        }
        
    def clear(self):
        self.tables = [{} for _ in range(self.num_tables)]
        self.store = {}
        self.hit_count = 0
        self.miss_count = 0
