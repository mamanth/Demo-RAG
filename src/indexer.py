import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB_MODEL = 'all-MiniLM-L6-v2'

def build_index(chunks, out_dir='../indexes'):
    os.makedirs(out_dir, exist_ok=True)
    model = SentenceTransformer(EMB_MODEL)
    texts = [c['text'] for c in chunks]
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize for cosine (inner product)
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, os.path.join(out_dir, 'faiss.index'))
    # write meta
    meta = [{'source_type':c['source_type'],'source_id':c['source_id'],'chunk_id':c['chunk_id'],'text':c['text']} for c in chunks]
    Path = __import__('pathlib').Path
    Path(out_dir).joinpath('meta.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print('Index built with', len(meta), 'items')

if __name__ == '__main__':
    from ingest import ingest_docs
    chunks = ingest_docs(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    build_index(chunks)
