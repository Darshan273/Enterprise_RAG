import logfire
from app.Agent.state import AgentState
from langchain_groq import ChatGroq
from config import settings

llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model=settings.GROQ_MODEL or "llama-3.3-70b-versatile",
    temperature=0.1
)

def planner_node(state: AgentState):
    """
    Planner node: Decide if query needs retrieval.
    
    Since guardrails already validated the query is on-topic,
    we just need to check if it's a simple greeting or needs research.
    """
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    # Only greetings skip retrieval
    simple_greetings = [
        "hello", "hi", "hey", "thanks", "thank you", 
        "bye", "goodbye", "good morning", "good evening"
    ]
    
    message_lower = user_message.lower().strip()
    
    if message_lower in simple_greetings:
        logfire.info("→ Conversational response (greeting)")
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally...",
            "plan": ["Type: Greeting", "Retrieval: Skipped"]
        }
    
    logfire.info(f"→ Technical/Research query: {user_message[:60]}")
    return {
        "current_query": user_message,
        "status": f"Research query - retrieving relevant papers...",
        "plan": ["Type: Technical", f"Query: {user_message[:50]}..."]
    }