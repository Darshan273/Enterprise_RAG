import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from app.Agent.graph import build_graph
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from config import settings
from langgraph.checkpoint.redis import AsyncRedisSaver
from app.database.connection import get_db, engine, Base
from app.database.schema import ConversationHistory
from langgraph.checkpoint.memory import MemorySaver
from app.guardrails import initialize_guardrails, validate_query


load_dotenv()
logfire.configure(token=settings.LOGFIRE_TOKEN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_guardrails()
    
    Base.metadata.create_all(bind=engine)
    redis_url = settings.REDIS_URL

    if redis_url:
        try:
            saver = AsyncRedisSaver.from_conn_string(redis_url)
            async with saver as checkpointer:
                await checkpointer.setup()
                app.state.graph = build_graph(checkpointer=checkpointer)
                yield
        except Exception as e:
            logfire.error(f"Redis Error: {e}")
            app.state.graph = build_graph(checkpointer=MemorySaver())
            yield
    else:
        app.state.graph = build_graph(checkpointer=MemorySaver())
        yield

app = FastAPI(title="enterprise RAG", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default_user"

@app.get("/")
def home():
    return {"message": "Enterprise RAG is live"}


@app.post("/query")
async def query(request: QueryRequest):
    question = request.question
    thread_id = request.thread_id
    
    is_valid, block_reason = await validate_query(question)
    
    if not is_valid:
        logfire.warning(f"Query blocked by guardrails")
        return {
            "question": question,
            "answer": block_reason or "Query rejected by security filters.",
            "thought_process": ["Guardrails validation failed"],
            "status": "blocked",
            "sources": []
        }
    
    logfire.info("Query passed guardrails, running planner...")
    
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "current_query": question,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing",
        "guardrails_passed": True,
        "guardrail_block_reason": None
    }

    config = {"configurable": {"thread_id": thread_id}}

    output = await app.state.graph.ainvoke(initial_state, config=config)
    
    return {
        "question": question,
        "answer": output["final_answer"],
        "thought_process": output["plan"],
        "status": output["status"],
        "sources": output["documents"]
    }

@app.post("/conversation/{thread_id}", status_code=status.HTTP_201_CREATED)
async def archive_conversation(thread_id: str, db=Depends(get_db)):
    """
    Retrieves the conversation state from Redis (LangGraph checkpointer)
    and saves the user/assistant message pairs into PostgreSQL.
    """
    if not hasattr(app.state, "graph") or app.state.graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph execution engine is not initialized."
        )

    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = await app.state.graph.aget_state(config)
        messages = state.values.get("messages", [])
        
        if not messages:
            return {"message": "No conversation history found for this session."}
        
        user_message = None
        archived_count = 0

        for msg in messages:
            # Check message format (dict or LangChain message class)
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                content = getattr(msg, "content", "")

            if role == "user":
                user_message = content
            elif role == "assistant" and user_message is not None:
                # Store paired message
                history_entry = ConversationHistory(
                    session_id=thread_id,
                    user_message=user_message,
                    ai_message=content
                )
                db.add(history_entry)
                user_message = None
                archived_count += 1
        
        db.commit()
        logfire.info(f"Archived {archived_count} message pairs to Postgres for session={thread_id}")
        return {"message": f"Successfully archived {archived_count} message pairs for session {thread_id}."}

    except Exception as e:
        db.rollback()
        logfire.error(f"Failed to archive conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive conversation: {str(e)}"
        )
