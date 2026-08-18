import openai
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy, default_retry_on

from content_agent.nodes.classify_decision import classify_decision
from content_agent.nodes.compliance import compliance_agent
from content_agent.nodes.creation import content_creation_agent
from content_agent.nodes.human_review import human_review
from content_agent.nodes.ideation import ideation_agent
from content_agent.observability import get_tracer, track_langgraph
from content_agent.routing import route_after_review
from content_agent.state import AgentState


def _openai_retry_on(exc: Exception) -> bool:
    """PRD §11.4 -- LangGraph's default_retry_on retries OpenAI's own
    exception types only by falling through its catch-all (they don't
    subclass ConnectionError/httpx.HTTPStatusError), which means it also
    retries AuthenticationError/BadRequestError the same way, wasting
    attempts on something that will never succeed. Retry only the
    genuinely transient OpenAI exceptions explicitly; let the rest of
    OpenAIError fail fast."""
    if isinstance(
        exc,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError),
    ):
        return True
    if isinstance(exc, openai.OpenAIError):
        return False
    return default_retry_on(exc)


_RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=_openai_retry_on)


def _approved(state: AgentState) -> dict:
    return {"final_content": state.get("draft_content")}


def _rejected(state: AgentState) -> dict:
    return {"final_content": None}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("ideation_agent", ideation_agent, retry_policy=_RETRY_POLICY)
    graph.add_node("content_creation_agent", content_creation_agent, retry_policy=_RETRY_POLICY)
    graph.add_node("compliance_agent", compliance_agent, retry_policy=_RETRY_POLICY)
    graph.add_node("human_review", human_review)
    graph.add_node("classify_decision", classify_decision, retry_policy=_RETRY_POLICY)
    graph.add_node("approved", _approved)
    graph.add_node("rejected", _rejected)

    graph.set_entry_point("ideation_agent")
    graph.add_edge("ideation_agent", "human_review")
    graph.add_edge("content_creation_agent", "human_review")
    graph.add_edge("compliance_agent", "human_review")
    graph.add_edge("human_review", "classify_decision")

    graph.add_conditional_edges(
        "classify_decision",
        route_after_review,
        {
            "human_review": "human_review",
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
    """Wraps the compiled graph with Opik tracing when OPIK_API_KEY is
    configured -- a true no-op otherwise (see content_agent.observability).
    Captures the full graph run plus every LangChain LLM call within it,
    including classify_decision now that it's a real node, grouped by
    LangGraph's own thread_id since that's already passed via
    config["configurable"] on every invoke()."""
    app = build_graph().compile(checkpointer=checkpointer or MemorySaver())
    tracer = get_tracer()
    if tracer is not None:
        app = track_langgraph(app, tracer)
    return app
