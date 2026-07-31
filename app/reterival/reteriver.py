import logfire
from config import settings
from app.ingestion.vectorDB import get_vector_store

def search_enterprise_knowledge(query: str, limit: int = 3):
    """
    Searches the Qdrant vector database for chunks relevant to the query.
    Returns a list of dictionaries with 'content' and 'metadata'.
    """
    with logfire.span("Qdrant Similarity Search", query=query, limit=limit):
        try:
            vector_store = get_vector_store()
            docs = vector_store.similarity_search(query, k=limit)
            results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            logfire.info(f"Retrieved {len(results)} raw documents from Qdrant")
            return results
        except Exception as e:
            logfire.error(f"Qdrant search failed: {e}")
            return []
