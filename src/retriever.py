import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB_MODEL = 'all-MiniLM-L6-v2'

class Retriever:
    def __init__(self, index_dir='../indexes', weights=None):
        # Make path relative to the file
        index_dir = Path(__file__).parent.parent / "indexes"  # points to Task-RAG/indexes
        faiss_index_path = index_dir / "faiss.index"
        meta_path = index_dir / "meta.json"

        # Load FAISS index
        print("Loading FAISS index from:", faiss_index_path)
        self.index = faiss.read_index(str(faiss_index_path))

        # Load metadata
        self.meta = json.loads(meta_path.read_text(encoding='utf8'))

        # Load embedding model
        self.model = SentenceTransformer(EMB_MODEL)

        # Set source weights
        self.weights = weights or {'product_doc': 0.5, 'blog': 0.3, 'forum': 0.2}

    def query(self, q, topk=10):
        # Encode query
        q_emb = self.model.encode(q, convert_to_numpy=True)
        faiss.normalize_L2(q_emb.reshape(1, -1))

        # Search FAISS index
        scores, ids = self.index.search(q_emb.reshape(1, -1), topk)

        res = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            m = self.meta[i]
            res.append({
                'score': float(s),
                'id': int(i),
                'source_type': m['source_type'],
                'source_id': m['source_id'],
                'chunk_id': m['chunk_id'],
                'text': m['text']
            })

        # Group by source and normalize per group, then apply weights
        grouped = {}
        for r in res:
            grouped.setdefault(r['source_type'], []).append(r)

        fused = []
        for src, items in grouped.items():
            arr = np.array([it['score'] for it in items], dtype=float)
            if arr.sum() == 0:
                norm = np.zeros_like(arr)
            else:
                exp = np.exp(arr - arr.max())
                norm = exp / exp.sum()
            w = self.weights.get(src, 0.1)
            for it, n in zip(items, norm):
                fused.append({**it, 'fused_score': float(n * w)})

        # Sort by fused score
        fused_sorted = sorted(fused, key=lambda x: x['fused_score'], reverse=True)
        return fused_sorted
