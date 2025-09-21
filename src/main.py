from src.ingest import ingest_docs
from src.retriever import Retriever
from src.reranker import Reranker
from src.utils import detect_contradictions

def run_demo():
    print("\n=================================================")
    print("Running Example Queries...")
    print("=================================================\n")

    example_queries = [
        "Can I generate reports automatically?",
        "How do I update the software?",
        "Can multiple users work simultaneously?",
        "What should I do if I encounter an error?",
    ]

    retriever = Retriever()
    reranker = Reranker()

    for query in example_queries:
        print(f"Q: {query}")
        retrieved = retriever.query(query)  # Use query() instead of retrieve()
        best_answer = reranker.rerank(query, retrieved)
        print(f"A: {best_answer}")

        # Detect contradictions using the retrieved chunks
        contradictions = detect_contradictions(query, retrieved)
        if contradictions:
            print("\nContradictions detected:")
            for c in contradictions:
                print(c)
        else:
            print("\nContradictions detected: None")
        print("\n" + "-"*50 + "\n")

    print("Demo complete.\n")


def interactive_chat():
    print("=================================================")
    print("Interactive Q&A Mode")
    print("Type 'exit' to quit.")
    print("=================================================\n")

    retriever = Retriever()
    reranker = Reranker()

    while True:
        query = input("Your Question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Thanks for trying ScriptChain RAG!")
            break

        retrieved = retriever.query(query)  # Use query() here too
        best_answer = reranker.rerank(query, retrieved)

        print(f"\nAnswer: {best_answer}\n")

        # Pass the retrieved chunks to detect contradictions
        contradictions = detect_contradictions(query, retrieved)
        if contradictions:
            print("Contradictions detected:")
            for c in contradictions:
                print(c)
        else:
            print("Contradictions detected: None")
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    ingest_docs()       # ingest once
    run_demo()          # run sample queries
    interactive_chat()  # allow user questions
