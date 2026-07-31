from typing import TypedDict, List, Annotated, Optional
import operator

class AgentState(TypedDict):
    
    messages: Annotated[List[dict], operator.add]
    current_query: str
    documents: List[str]
    plan: List[str]
    status: str
    final_answer: str
    guardrails_passed: bool 
    guardrail_block_reason: Optional[str]  