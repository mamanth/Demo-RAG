import os, json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pathlib import Path

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding='utf8')

def chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    words = text.split()
    chunks = []
    i = 0
    # approximate by words to avoid tokenizers
    while i < len(words):
        chunk = " ".join(words[i:i+max_chars])
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks

def ingest_docs(base_dir: str = '../docs') -> List[Dict]:
    base = Path(base_dir).resolve()
    out = []
    # product docs - md
    for p in (base / 'product_docs').glob('*.md'):
        text = read_text_file(str(p))
        chunks = chunk_text(text, max_chars=200, overlap=40)
        for i,c in enumerate(chunks):
            out.append({'source_type':'product_doc','source_id':p.name,'chunk_id':f'{p.stem}_c{i}','text':c})
    # forums - txt
    for p in (base / 'forums').glob('*.txt'):
        text = read_text_file(str(p))
        chunks = chunk_text(text, max_chars=300, overlap=60)
        for i,c in enumerate(chunks):
            out.append({'source_type':'forum','source_id':p.name,'chunk_id':f'{p.stem}_c{i}','text':c})
    # blogs - md
    for p in (base / 'blogs').glob('*.md'):
        text = read_text_file(str(p))
        chunks = chunk_text(text, max_chars=250, overlap=50)
        for i,c in enumerate(chunks):
            out.append({'source_type':'blog','source_id':p.name,'chunk_id':f'{p.stem}_c{i}','text':c})
    print(f'Ingested {len(out)} chunks total')
    return out

if __name__ == '__main__':
    chunks = ingest_docs(os.path.join(os.path.dirname(__file__), '..', 'docs'))
    print('Sample chunk:', chunks[0]['text'][:200])
