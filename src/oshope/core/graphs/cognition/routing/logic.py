from oshope.core.states.oshope_state import OSHopeState
from langgraph.graph import END
from oshope.utils.logger import get_logger
from oshope.config.config import (
    QUERY_CLARIFICATION_NODE,
    QUERY_CLASSIFICATION_NODE,
    CLARIFICATION_NODE_MAX_ATTEMPTS,
)

logger = get_logger(__name__)


def route_query_after_starting(state: OSHopeState) -> str:
    """
    Determines the next node to route the query to after the initial start node.

    Args:
        state (OSHopeState): The current state of the OS-HOPE.

    Returns:
        str: The name of the node to route the query to.
    """
    if state.query_classification.requires_follow_up:
        return QUERY_CLARIFICATION_NODE
    return QUERY_CLASSIFICATION_NODE


def route_query_after_classification(state: OSHopeState) -> str:
    """
    Determines the next node to route the query to based on the current state of the OS-HOPE.

    Args:
        state (OSHopeState): The current state of the OS-HOPE, including the query classification results.

    Returns:
        str: The name of the node to route the query to.
    """
    if (
        state.query_classification.requires_follow_up
        and state.clarification_attempts < CLARIFICATION_NODE_MAX_ATTEMPTS
    ):
        logger.info("Routing to query clarification node due to follow-up requirement.")
        return QUERY_CLARIFICATION_NODE

    return END
