from transformers import pipeline

# Load NLI model for contradiction detection
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

def detect_contradictions(query, retrieved_chunks):
    """
    Compare the query against retrieved chunks to see if any contradictions exist.
    """
    results = []
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        if not text:
            continue

        try:
            output = nli_model(
                {"text": query, "text_pair": text},
                truncation=True
            )
            results.append(output[0])
        except Exception as e:
            results.append({"label": "ERROR", "score": 0.0, "error": str(e)})

    return results
