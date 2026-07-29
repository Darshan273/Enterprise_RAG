from langgraph.graph import StateGraph, END
from app.Agent.state import AgentState
from app.Agent.nodes.planner import planner_node
from app.Agent.nodes.retriever import retrieve_node
from app.Agent.nodes.responder import generate_node
from langgraph.checkpoint.redis import AsyncRedisSaver

def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    def route_planner(state: AgentState):
        if state["current_query"] == "CONVERSATIONAL":
            return "responder"
        return "retriever"

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {
            "retriever": "retriever",
            "responder": "responder"
        }
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    rag_agent = workflow.compile(checkpointer=checkpointer)
    return rag_agent
