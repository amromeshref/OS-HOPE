from oshope.core.states.oshope_state import (
    OSHopeState,
    StepResolverState,
    CommandExecution,
    CommandErrorHandlerState,
    InformationResponse,
    SummarizerState,
)
from oshope.utils.logger import get_logger
import threading

logger = get_logger(__name__)


state_lock = threading.Lock()


def update_state(
    state: OSHopeState,
    response=None,
    executed_step_summary=None,
    step_index: int = None,
    step_status: str = None,
):

    with state_lock:
        logger.info("Updating the state")

        if isinstance(response, StepResolverState):
            state.planning.plan_steps[step_index].steps_resolver.append(response)
            state.steps_resolver.append(response)

        elif isinstance(response, CommandExecution):
            state.planning.plan_steps[step_index].command_executions.append(response)
            state.command_executions.append(response)

        elif isinstance(
            response,
            CommandErrorHandlerState,
        ):
            state.planning.plan_steps[step_index].command_error_handlers.append(
                response
            )
            state.planning.plan_steps[step_index].num_error_executions += 1
            state.command_error_handlers.append(response)

        elif isinstance(response, InformationResponse):
            state.generated_information_responses.append(response)

        elif isinstance(response, str):
            state.generated_final_response = response

        elif isinstance(response, SummarizerState):
            state.memory_extraction = response
            state.past_session_summaries.append(response.session_summary)

        else:
            logger.warning(f"Unknown response type: " f"{type(response)}")

        if executed_step_summary:
            state.executed_steps.append(executed_step_summary)

        if step_status:
            state.planning.plan_steps[step_index].status = step_status

        logger.info(f"State updated successfully")
