# Enterprise RAG

Enterprise-grade Retrieval-Augmented Generation (RAG) service built with FastAPI + LangGraph, featuring guardrails, reranking, Redis checkpointing, and PostgreSQL long-term memory.

## What This Project Includes

- Multi-step agent workflow with planner -> retriever -> responder nodes
- Semantic retrieval from Qdrant vector database
- Embedding pipeline using Jina Embeddings
- Reranking using FlashRank for better final context quality
- Safety and policy filtering with NeMo Guardrails
- Conversation state checkpointing with Redis via LangGraph checkpointer
- Long-term memory archival in PostgreSQL

## Technologies Used

- API and server
	- FastAPI
	- Uvicorn

- Agent orchestration
	- LangGraph
	- LangChain

- Retrieval and vector search
	- Qdrant (`qdrant-client`)
	- `langchain-qdrant`

- Embeddings
	- Jina Embeddings (`langchain-community` + Jina API)
	- `sentence-transformers` (included for embedding workflows)

- Reranking
	- FlashRank (`flashrank`)

- Guardrails and safety
	- NeMo Guardrails (`nemoguardrails==0.23.0`)
	- Groq LLM (`langchain-groq`) for safety classification

- Memory and persistence
	- Redis (`redis`) for short-term conversation checkpointing
	- LangGraph Redis checkpointer (`langgraph-checkpoint-redis`)
	- PostgreSQL (`sqlalchemy` + `psycopg2-binary`) for long-term memory storage

- Observability
	- Logfire
	- Loguru

## Core Architecture

```mermaid
flowchart LR
		U[User Query] --> G[NeMo Guardrails]
		G -->|Pass| P[Planner Node]
		G -->|Blocked| B[Refusal Response]

		P -->|Conversational| R[Responder Node]
		P -->|Needs Retrieval| T[Retriever Node]

		T --> Q[Qdrant Vector Search]
		Q --> RR[FlashRank Reranking]
		RR --> R

		R --> A[Final Answer]

		subgraph Memory
			C[Redis Checkpointer]
			L[PostgreSQL Long-Term Memory]
		end

		P -. state .-> C
		T -. state .-> C
		R -. state .-> C
		A -. archive endpoint .-> L
```

## How Reranking Works

After vector retrieval from Qdrant, the retrieved chunks are reranked with FlashRank relative to the user query. This improves answer quality by promoting the most relevant passages before response generation.

## How Embeddings Work

Document chunks are converted into vectors using Jina Embeddings (`jina-embeddings-v2-base-en`) and stored in Qdrant. Query vectors are matched against these stored vectors during retrieval.

## NeMo Guardrails Usage

The API validates each input query through a NeMo Guardrails service before running the agent graph.

- Jailbreak and off-topic detection are handled by a guardrails safety classifier.
- Blocked requests return controlled refusal responses.
- Allowed requests continue into the planner/retriever/responder flow.

## Redis Checkpointer Usage

LangGraph is compiled with a Redis checkpointer (`AsyncRedisSaver`) when `REDIS_URL` is available.

- Stores short-term execution state and message history per `thread_id`
- Enables resumable, session-aware conversations
- Falls back to in-memory checkpointing if Redis is unavailable

## Long-Term Memory with PostgreSQL

This project uses PostgreSQL as long-term memory for conversation history.

- Runtime conversation state remains in checkpoint storage
- `/conversation/{thread_id}` endpoint archives user/assistant message pairs
- Data is stored in the `conversational_history` table via SQLAlchemy

## Project Structure

```text
app/
	Agent/
		graph.py
		state.py
		nodes/
			planner.py
			retriever.py
			responder.py
	ingestion/
		chunking.py
		embedding.py
		process.py
		vectorDB.py
	reterival/
		rerank.py
		reteriver.py
	guardrails/
		config.yml
		prompts.yml
		service.py
	database/
		connection.py
		schema.py
main.py
config.py
```

## Environment Variables

Create a `.env` file with at least:

```env
JINA_EMBEDDING_API_KEY=your_jina_key
QDRANT_CLUSTER_ENDPOINT=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION_NAME=arxiv_collection

GROQ_API_KEY=your_groq_key
NEMOGUARDRAILS_CONFIG_PATH=app/guardrails

POSTGRESQL_URL=postgresql+psycopg2://user:password@host:5432/dbname
REDIS_URL=redis://localhost:6379/0

LOGFIRE_TOKEN=your_logfire_token
NVIDIA_API_KEY=optional_nvidia_key
```

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

API runs at `http://127.0.0.1:8000`.

## Key Endpoints

- `GET /` - health/welcome response
- `POST /query` - guardrailed RAG query endpoint
- `POST /conversation/{thread_id}` - archive session history to PostgreSQL

## Summary

This repository combines:

- Retrieval + embeddings (Qdrant + Jina)
- Reranking (FlashRank)
- Safety (NeMo Guardrails)
- Stateful sessions (Redis checkpointer)
- Long-term memory (PostgreSQL)

to deliver a production-oriented Enterprise RAG pipeline.
