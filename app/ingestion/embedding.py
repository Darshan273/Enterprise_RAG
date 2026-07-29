from langchain_community.embeddings import JinaEmbeddings
import logfire
from config import settings

def embed(chunk):
    with logfire.span("embedding"):
        logfire.info("Embedding the document")
        embedding = JinaEmbeddings(
            jina_api_key=settings.JINA_EMBEDDING_API_KEY,
            model_name="jina-embeddings-v2-base-en",
        )
        logfire.info("Embedded the document")
    return embedding.embed_documents(chunk)