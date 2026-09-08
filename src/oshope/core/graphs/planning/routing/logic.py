from oshope.core.states.oshope_state import OSHopeState
from langgraph.graph import END
from oshope.utils.logger import get_logger
from oshope.config.config import (
    USER_VALIDATION_NODE,
    PLANNING_NODE,
)
from langgraph.graph import END

logger = get_logger(__name__)


def route_after_starting(state: OSHopeState) -> str:
    """
    Routes the flow after the START node based on whether user validation is required.
    If user validation is required, it routes to the USER_VALIDATION_NODE. Otherwise, it routes to the PLANNING_NODE to generate the execution/information plan.
    Args:
        state (OSHopeState): The current state of the OS-HOPE, which includes information about the user's query and any preliminary processing results that may indicate whether user validation is needed.
    Returns:
        str: The next node to route to, either USER_VALIDATION_NODE or PLANNING_NODE.
    """
    if state.plan_presented:
        return USER_VALIDATION_NODE
    return PLANNING_NODE


def route_after_planning(state: OSHopeState) -> str:
    """
    Routes the flow after the planning node based on the query classification and whether follow-up is required.
    If the query is classified as "information" or if follow-up is required, it routes to the END node.
    Otherwise, it routes to the USER_VALIDATION_NODE for user validation of the generated plan.
    Args:
        state (OSHopeState): The current state of the OS-HOPE, which includes information about the user's query, the results of query classification, and the planning results.
    Returns:
        str: The next node to route to, either END or USER_VALIDATION_NODE.
    """
    if (
        state.query_classification.query_type == "information"
        or state.planning.requires_follow_up
    ):
        return END
    return USER_VALIDATION_NODE
