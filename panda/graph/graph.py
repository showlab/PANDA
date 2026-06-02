"""PANDA Agent graph — LangGraph workflow with perception/planning/analysis/reflection."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from panda.state import VadAgentState


def should_continue_analysis(
    state: VadAgentState,
) -> Literal["continue", "reflection", "end"]:
    """Route analysis output: continue to next window, reflect, or end."""
    if state.get("window_index", 0) >= len(state.get("all_windows", [])):
        return END
    if state.get("retry_count", 0) > 3:
        return "continue"
    results = state.get("analysis_result", [])
    if results and results[-1].get("status") == "insufficient":
        return "reflection"
    return "continue"


def build_graph(llm=None):
    """Build and compile the PANDA LangGraph workflow.

    Args:
        llm: Optional LangChain LLM (not currently used by agent nodes but
             kept for API compatibility).

    Returns:
        Compiled LangGraph graph.
    """
    from panda.agents.analysis import analysis_agent
    from panda.agents.perception import perception_agent
    from panda.agents.planning import planning_agent
    from panda.agents.reflection import reflection_agent

    workflow = StateGraph(VadAgentState)

    workflow.add_node("perception", perception_agent)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("reflection", reflection_agent)

    workflow.add_edge(START, "perception")
    workflow.add_edge("perception", "planning")
    workflow.add_edge("planning", "analysis")

    workflow.add_conditional_edges(
        "analysis",
        should_continue_analysis,
        {
            "continue": "analysis",
            "reflection": "reflection",
            END: END,
        },
    )
    workflow.add_edge("reflection", "analysis")

    return workflow.compile()
