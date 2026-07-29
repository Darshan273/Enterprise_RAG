import time
import logfire
from flashrank import Ranker, RerankRequest

# Initialize the FlashRank Ranker once globally to avoid reloading model
try:
    with logfire.span("Initializing FlashRank"):
        # ms-marco-MiniLM-L-12-v2 is the default model
        ranker = Ranker()
except Exception as e:
    logfire.error(f"Failed to initialize FlashRank: {e}")
    ranker = None

def rerank_documents(query: str, doc_contents: list[str], top_n: int = 20) -> list[str]:
    """
    Reranks document chunks using FlashRank relative to the query.
    Returns the top_n most relevant chunks.
    """
    if not ranker or not doc_contents:
        logfire.warning("FlashRank ranker is uninitialized or document list is empty. Returning raw chunks.")
        return doc_contents[:top_n]

    with logfire.span("FlashRank Reranking", query=query, doc_count=len(doc_contents), top_n=top_n):
        try:
            passages = [{"id": i, "text": doc} for i, doc in enumerate(doc_contents)]
            rerank_request = RerankRequest(query=query, passages=passages)
            
            start_time = time.time()
            results = ranker.rerank(rerank_request)
            duration = time.time() - start_time
            
            logfire.info(f"Reranked {len(doc_contents)} docs in {duration:.4f}s")
            
            # Extract and return the reranked texts
            reranked_texts = [res["text"] for res in results[:top_n]]
            return reranked_texts
        except Exception as e:
            logfire.error(f"Error during FlashRank reranking: {e}")
            return doc_contents[:top_n]