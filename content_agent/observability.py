import os

from dotenv import load_dotenv

load_dotenv()

OPIK_ENABLED = bool(os.environ.get("OPIK_API_KEY"))

if OPIK_ENABLED:
    from opik import track
    from opik.integrations.langchain import OpikTracer, track_langgraph
else:
    # opik.track calls out over the network the moment a decorated function
    # runs, even with no API key configured (it just gets a 401) -- that
    # would inject real network calls into the offline test suite and any
    # environment without Opik set up. Fall back to true no-ops instead.
    def track(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

    def track_langgraph(app, tracer):
        return app

    OpikTracer = None


def get_tracer():
    if not OPIK_ENABLED:
        return None
    return OpikTracer(project_name=os.environ.get("OPIK_PROJECT_NAME", "content-agent"))
