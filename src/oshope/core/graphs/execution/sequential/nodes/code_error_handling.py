from oshope.core.states.oshope_state import (
    OSHopeState,
    CommandErrorHandlerState,
)
from oshope.prompts.code_error_handling import (
    get_code_error_handling_sys_prompt,
    get_first_human_message,
    get_second_human_message,
)
from oshope.utils.logger import get_logger
from oshope.core.models.main import LLMModel

logger = get_logger(__name__)


def code_error_handling_node(state: OSHopeState) -> OSHopeState:
    """
    Node responsible for handling errors that occur during code/command execution as part of the execution graph.
    """
    logger.info("Starting code error handling node.")

    sys_prompt: str = get_code_error_handling_sys_prompt()
    llm_model = LLMModel()

    if state.num_error_executions == 0:
        human_message = get_first_human_message(state)
    else:
        human_message = get_second_human_message(state)

    response: CommandErrorHandlerState = llm_model.generate_response(
        system_message=sys_prompt,
        human_message=human_message,
        structured_output=CommandErrorHandlerState,
    )

    state.command_error_handlers.append(response)
    state.num_error_executions += 1
    logger.info("Completed code error handling node.")

    if not response.can_recover:
        state.executed_steps.append(
            f"Could not recover from the error in step index {state.current_step_index} and description {state.planning.plan_steps[state.current_step_index].step_details.description} after {state.num_error_executions} recovery attempts. Reason: {response.recovery_reasoning}"
        )
        logger.info("Cannot recover from the error. Moving to the next step.")

    return state
