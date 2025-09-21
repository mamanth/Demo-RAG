# src/reranker.py
class Reranker:
    """
    Simple reranker that picks the chunk with the highest final_score
    and returns its text.
    """
    def rerank(self, query, retrieved):
        if not retrieved:
            return "No relevant information found."
        # Pick the chunk with the highest final_score
        best_chunk = max(retrieved, key=lambda x: x.get('final_score', 0))
        return best_chunk['text']
