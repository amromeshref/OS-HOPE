from oshope.core.states.oshope_state import OSHopeState, QueryClarificationState
from oshope.prompts.query_clarification import (
    get_query_clarification_sys_prompt,
    get_first_human_message,
    get_second_human_message,
)
from oshope.utils.logger import get_logger
from oshope.core.models.main import LLMModel

logger = get_logger(__name__)


def query_clarification_node(state: OSHopeState) -> OSHopeState:
    """
    Generates a clarification response using an LLM based on the original query and its classification.
    Args:
        state (OSHopeState): The current state of the OS-HOPE, including the original query and its classification.
    Returns:
        OSHopeState: The updated state with the generated response for clarification.
    """
    logger.info("Starting query clarification node.")
    state.clarification_attempts += 1

    llm_model = LLMModel()
    sys_prompt = get_query_clarification_sys_prompt()

    if state.planning.requires_follow_up:
        human_message = get_second_human_message(state)
    else:
        human_message = get_first_human_message(state)

    # Generate the classification response from the LLM
    response: QueryClarificationState = llm_model.generate_response(
        system_message=sys_prompt,
        human_message=human_message,
        structured_output=QueryClarificationState,
    )

    # TODO: Implement parsing logic

    state.query_clarification = response
    logger.info("Completed query clarification node.")

    if not state.query_clarification.is_clarification_needed:
        state.finalized_enhanced_query = (
            state.query_clarification.finalized_enhanced_query
        )

    return state

    # Parse the classification response and update the state
    # (This is a placeholder for actual parsing logic, which would depend on the format of the LLM's response)
    # For example, you might expect a JSON response that can be directly converted to ClassificationState
