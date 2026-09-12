from oshope.core.states.oshope_state import OSHopeState, InformationResponse, Step
from oshope.prompts.information_generation import (
    get_information_generation_sys_prompt,
    get_human_message,
)
from oshope.utils.logger import get_logger
from oshope.core.models.main import LLMModel
from oshope.tools.retrieve_execution_details import retrieve_execution_details
from oshope.tools.retrieve_information_details import retrieve_information_details

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


def information_generation_node(state: OSHopeState) -> OSHopeState:
    logger.info("Starting information generation node.")

    # if len(state.planning.information_steps) == 0:
    #     logger.info("No information query provided")
    #     logger.info("Completed information generation node.")
    #     return state

    llm_model = LLMModel()
    sys_prompt = get_information_generation_sys_prompt()

    step_index = state.current_step_index

    information_response = InformationResponse()
    dependency_outputs_str = ""

    if state.steps_resolver_active:
        current_resolving_step_index = state.current_resolving_step_index
        current_resolving_step = state.steps_resolver[-1].resolved_steps[
            current_resolving_step_index
        ]
        information_response.query = current_resolving_step.step_details.description
        if current_resolving_step.dependencies_required:
            dependency_outputs_str = retrieve_dependency_outputs(
                state, current_resolving_step
            )
        state.current_resolving_step_index += 1
    else:
        current_step = state.planning.plan_steps[step_index]
        information_response.query = current_step.step_details.description
        if current_step.dependencies_required:
            dependency_outputs_str = retrieve_dependency_outputs(state, current_step)

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
    information_response.dependency_outputs = dependency_outputs_str
    state.generated_information_responses.append(information_response)
    state.executed_steps.append(
        f"The information step involving the query '{information_response.query}' is done."
    )
    # state.current_step_index += 1

    # state.steps_done_indicies.append(state.execution_orchestrator[-1].next_step_index)

    # save_information_responses(state.generated_information_responses)

    if state.steps_resolver_active:
        current_resolving_step_index = state.current_resolving_step_index
        total_resolved_steps = len(state.steps_resolver[-1].resolved_steps)

        if current_resolving_step_index >= total_resolved_steps:
            logger.info("All resolved steps have been executed.")
            state.steps_resolver_active = False
            state.current_resolving_step_index = 0
            state.current_step_index += 1
    else:
        state.current_step_index += 1

    state.generated_information_responses_status = "completed"
    logger.info("Completed information generation node.")

    return state
