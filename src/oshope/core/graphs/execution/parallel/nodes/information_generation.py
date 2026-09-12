from oshope.core.states.oshope_state import OSHopeState, InformationResponse, Step
from oshope.prompts.information_generation import (
    get_information_generation_sys_prompt,
    get_human_message,
)
from oshope.utils.logger import get_logger
from oshope.core.models.main import LLMModel
from oshope.tools.retrieve_execution_details import retrieve_execution_details
from oshope.tools.retrieve_information_details import retrieve_information_details
from oshope.core.graphs.execution.parallel.state_manager import update_state

logger = get_logger(__name__)


def retrieve_dependency_outputs(state: OSHopeState, current_step: Step) -> str:
    """
    Retrieve the outputs of the dependencies for the current step.
    Args:
        state: OSHopeState
        current_step: The current step for which to retrieve dependency outputs
    Returns:
        str: A string representation of the dependency outputs.
    """
    dependency_outputs = []

    for dep_idx in current_step.dependency_step_indices:
        dep_output = None
        if state.planning.plan_steps[dep_idx].step_type == "command":
            dep_output = retrieve_execution_details(state, dep_idx)
        elif state.planning.plan_steps[dep_idx].step_type == "information":
            dep_output = retrieve_information_details(state, dep_idx)
        dependency_outputs.append(dep_output)

    return "\n".join(dependency_outputs)


def information_generation_node(state: OSHopeState, step: Step):
    logger.info("Starting information generation node.")

    step_index = step.step_index

    llm_model = LLMModel()
    sys_prompt = get_information_generation_sys_prompt()

    information_response = InformationResponse()
    dependency_outputs_str = ""

    information_response.query = step.step_details.description
    if step.dependencies_required:
        dependency_outputs_str = retrieve_dependency_outputs(state, step)

    human_message = get_human_message(
        state, information_response.query, dependency_outputs_str
    )

    response: str = llm_model.generate_response(
        human_message=human_message,
        system_message=sys_prompt,
        structured_output=None,
    )

    # TODO: Add parsing logic here

    information_response.answer = response
    information_response.step_index = step_index
    executed_step_summary = f"The information step involving the query '{information_response.query}' is done."
    # state.current_step_index += 1

    # state.steps_done_indicies.append(state.execution_orchestrator[-1].next_step_index)

    # save_information_responses(state.generated_information_responses)

    update_state(
        state=state,
        response=information_response,
        executed_step_summary=executed_step_summary,
    )

    logger.info("Completed information generation node.")

    return {
        "step_index": step_index,
        "success": True,
    }
