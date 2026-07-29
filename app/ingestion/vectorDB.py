import logfire
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from config import settings
from langchain_community.embeddings import JinaEmbeddings

embeddings = JinaEmbeddings(
    jina_api_key=settings.JINA_EMBEDDING_API_KEY,
    model_name="jina-embeddings-v2-base-en",
)

def get_qdrant_client():
    return QdrantClient(
        url=settings.QDRANT_CLUSTER_ENDPOINT,
        api_key=settings.QDRANT_API_KEY,
    )

def get_vector_store():
    with logfire.span("get_vector_store"):
        logfire.info("Initializing QdrantVectorStore")
        client = get_qdrant_client()
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )
        return vector_store

def insert_documents(docs):
    with logfire.span("insert_documents"):
        logfire.info(f"Inserting {len(docs)} document chunks into Qdrant")
        vector_store = get_vector_store()
        
        vector_store.add_documents(docs)
        logfire.info("Insertion complete")

def wipe_collection():
    with logfire.span("wipe_collection"):
        logfire.info("Wiping the Qdrant collection")
        client = get_qdrant_client()
        
        if client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            client.delete_collection(collection_name=settings.QDRANT_COLLECTION_NAME)
            logfire.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' deleted.")
        else:
            logfire.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' does not exist.")
        
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logfire.info(f"Recreated empty collection '{settings.QDRANT_COLLECTION_NAME}'.")
