import logfire
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from app.Agent.graph import build_graph
from app.guardrails import initialize_rails, guard
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from config import settings
from langgraph.checkpoint.redis import AsyncRedisSaver
from app.database.connection import get_db, engine, Base
from app.database.schema import ConversationHistory
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
logfire.configure(token=settings.LOGFIRE_TOKEN)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_rails()  
    try:
        with logfire.span("Initializing Database Tables"):
            Base.metadata.create_all(bind=engine)
            logfire.info("PostgreSQL tables successfully initialized.")
    except Exception as e:
        logfire.error(f"PostgreSQL initialization failed: {e}")

    redis_url = settings.REDIS_URL
    if redis_url:
        try:
            saver_context = AsyncRedisSaver.from_conn_string(redis_url)
            async with saver_context as checkpointer:
                await checkpointer.setup()
                app.state.graph = build_graph(checkpointer=checkpointer)
                yield
        except Exception as redis_err:
            logfire.error(f"Redis connection failed: {redis_err}. Falling back to MemorySaver.")
            app.state.graph = build_graph(checkpointer=MemorySaver())
            yield
    else:
        logfire.warning("REDIS_URL not configured. Running without checkpointer/memory storage.")
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
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    Evaluates query against input rails first.
    """
    question = request.question
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "current_query": question,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Check input guardrails
        rail_fired, rail_response = guard(question)

        if rail_fired:
            logfire.info(f"Request blocked by guardrails | thread={thread_id}")
            return {
                "question": question,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Invoke compiled LangGraph workflow from app state
        final_output = await app.state.graph.ainvoke(initial_state, config=config)
        
        return {
            "question": question,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"Backend Execution Failed: {e}")
        return {
            "question": question,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }

@app.post("/conversation/{session_id}", status_code=status.HTTP_201_CREATED)
async def archive_conversation(session_id: str, db=Depends(get_db)):
    """
    Retrieves the conversation state from Redis (LangGraph checkpointer)
    and saves the user/assistant message pairs into PostgreSQL.
    """
    if not hasattr(app.state, "graph") or app.state.graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph execution engine is not initialized."
        )

    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Fetch current state from memory checkpointer
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
                    session_id=session_id,
                    user_message=user_message,
                    ai_message=content
                )
                db.add(history_entry)
                user_message = None
                archived_count += 1
        
        db.commit()
        logfire.info(f"Archived {archived_count} message pairs to Postgres for session={session_id}")
        return {"message": f"Successfully archived {archived_count} message pairs for session {session_id}."}

    except Exception as e:
        db.rollback()
        logfire.error(f"Failed to archive conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive conversation: {str(e)}"
        )
