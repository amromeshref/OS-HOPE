from oshope.core.states.oshope_state import OSHopeState, UserValidationState
from oshope.core.models.main import LLMModel
from oshope.prompts.user_validation import (
    get_user_validation_sys_prompt,
    get_human_message,
)
from oshope.utils.logger import get_logger

logger = get_logger(__name__)


def user_validation_node(state: OSHopeState) -> OSHopeState:
    """
    Validates the generated execution/information plan with the user to ensure it aligns with their original query and intentions.
    Args:
        state (OSHopeState): The current state of the OS-HOPE, including the generated execution/information plan.
    Returns:
        OSHopeState: The updated state after user validation, which may include any adjustments to the plan based on user feedback.
    """
    logger.info("Starting user validation node.")

    state.user_validation_attempts += 1

    llm_model = LLMModel()
    sys_prompt = get_user_validation_sys_prompt()
    human_message = get_human_message(state)

    while True:
        try:
            response: UserValidationState = llm_model.generate_response(
                system_message=sys_prompt,
                human_message=human_message,
                structured_output=UserValidationState,
            )
            break  # Exit the loop if response is successfully generated
        except Exception as e:
            error_message = str(e)

            # Detect tool call failure
            if "tool" in error_message.lower():
                logger.warning("Tool call failed during user validation. Retrying...")
                human_message += "\nNote: The previous response encountered an error related to tool calls. Please ensure that the response adheres to the expected format and does not include any tool calls."
                continue  # Retry the user validation node

    # TODO: Implement parsing logic

    state.user_validation = response
    state.user_validation_status = "completed"

    logger.info("User validation node completed.")

    return state
