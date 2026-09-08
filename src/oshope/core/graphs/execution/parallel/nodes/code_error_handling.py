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
from oshope.core.graphs.execution.parallel.state_manager import update_state

logger = get_logger(__name__)


def code_error_handling_node(state: OSHopeState, step_index: int):
    """
    Node responsible for handling errors that occur during code/command execution as part of the execution graph.
    """
    logger.info("Starting code error handling node.")

    sys_prompt: str = get_code_error_handling_sys_prompt()
    llm_model = LLMModel()

    if state.planning.plan_steps[step_index].num_error_executions == 0:
        human_message = get_first_human_message(
            state, step_index=step_index
        )
    else:
        human_message = get_second_human_message(
            state, step_index=step_index
        )

    response: CommandErrorHandlerState = llm_model.generate_response(
        system_message=sys_prompt,
        human_message=human_message,
        structured_output=CommandErrorHandlerState,
    )

    executed_step_summary = None

    if not response.can_recover:
        executed_step_summary = f"Could not recover from the error in step index {step_index} and description {state.planning.plan_steps[step_index].step_details.description} after {state.planning.plan_steps[step_index].num_error_executions} recovery attempts. Reason: {response.recovery_reasoning}"
        logger.info("Cannot recover from the error. Moving to the next step.")

    update_state(
        state=state,
        response=response,
        executed_step_summary=executed_step_summary,
        step_index=step_index,
    )

    logger.info("Completed code error handling node.")

    return {
        "step_index": step_index,
        "success": True,
    }
