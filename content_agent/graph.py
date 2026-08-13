from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from content_agent.nodes.compliance import compliance_agent
from content_agent.nodes.creation import content_creation_agent
from content_agent.nodes.human_review import human_review
from content_agent.nodes.ideation import ideation_agent
from content_agent.routing import route_after_review
from content_agent.state import AgentState


def _approved(state: AgentState) -> dict:
    return {"final_content": state.get("draft_content")}


def _rejected(state: AgentState) -> dict:
    return {"final_content": None}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ideation_agent", ideation_agent)
    graph.add_node("content_creation_agent", content_creation_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("human_review", human_review)
    graph.add_node("approved", _approved)
    graph.add_node("rejected", _rejected)

    graph.set_entry_point("ideation_agent")
    graph.add_edge("ideation_agent", "human_review")
    graph.add_edge("content_creation_agent", "human_review")
    graph.add_edge("compliance_agent", "human_review")

    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "ideation_agent": "ideation_agent",
            "content_creation_agent": "content_creation_agent",
            "compliance_agent": "compliance_agent",
            "approved": "approved",
            "rejected": "rejected",
        },
    )
    graph.add_edge("approved", END)
    graph.add_edge("rejected", END)

    return graph


def compile_app(checkpointer=None):
    return build_graph().compile(checkpointer=checkpointer or MemorySaver())
