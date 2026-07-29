import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    JINA_EMBEDDING_KEY = os.getenv("JINA_EMBEDDING_KEY") or os.getenv("JINA_EMBEDDING")
    JINA_EMBEDDING_API_KEY = JINA_EMBEDDING_KEY

    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL")
    QDRANT_URL = QDRANT_CLUSTER_ENDPOINT
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "arxiv_collection")
    QDRANT_COLLECTION_NAME = QDRANT_COLLECTION

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

    POSTGRESQL_URL=os.getenv("POSTGRESQL_URL")
    REDIS_URL=os.getenv("REDIS_URL")

settings = Settings()