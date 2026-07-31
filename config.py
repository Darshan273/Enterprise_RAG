import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    JINA_EMBEDDING_API_KEY = os.getenv("JINA_EMBEDDING_API_KEY")

    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "arxiv_collection")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

    POSTGRESQL_URL = os.getenv("POSTGRESQL_URL")
    REDIS_URL = os.getenv("REDIS_URL")

    NEMOGUARDRAILS_CONFIG_PATH = os.getenv("NEMOGUARDRAILS_CONFIG_PATH", "app/guardrails")

settings = Settings()