from oshope.core.states.oshope_state import OSHopeState, StepResolverState, Step
from oshope.prompts.step_resolver import get_step_resolver_sys_prompt, get_human_message
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
    Returns:
        str: A string representation of the dependency outputs.
    """
    dependency_outputs = []

    for dep_idx in current_step.dependency_step_indices:
        dep_output = None
        if state.planning.plan_steps[dep_idx].step_type == "command":
            dep_output = retrieve_execution_details(
                state, dep_idx
            )
        elif state.planning.plan_steps[dep_idx].step_type == "information":
            dep_output = retrieve_information_details(state, dep_idx)
        dependency_outputs.append(dep_output)

    return "\n".join(dependency_outputs)


def step_resolver_node(state: OSHopeState, step: Step) -> StepResolverState:
    """
    Resolve the current step by replacing placeholders with actual values from dependencies.
    """
    logger.info("Starting step resolver node.")

    step_index = step.step_index

    if step.step_type == "command":
        if len(step.step_details.input_variables) == 0:
            logger.info("No input variables for this command step. Skipping step resolver.")
            return StepResolverState(
                is_resolution_successful=True,
                resolved_steps=[step],
                resolution_reasoning="No input variables to resolve.",
            )

    llm_model = LLMModel()
    sys_prompt = get_step_resolver_sys_prompt()

    print("=" * 90)
    print(f"Current step index in step resolver: {step_index}")
    print("=" * 90)

    dependency_outputs_str = retrieve_dependency_outputs(state, step)

    human_message = get_human_message(step, dependency_outputs_str)

    response: StepResolverState = llm_model.generate_response(
        system_message=sys_prompt,
        human_message=human_message,
        structured_output=StepResolverState,
    )

    # TODO: Add parsing logic

    # TODO: implement a function to update the state.(wait and lock)

    executed_step_summary = None

    if not response.is_resolution_successful:
        executed_step_summary = f"Could not execute the step whose index is {step_index} and description is {step.description} due to the following reason: {response.resolution_reasoning}"
        logger.info("Step resolution failed. Moving to the next step.")

    update_state(
        state=state,
        response=response,
        executed_step_summary=executed_step_summary,
        step_index=step_index,
    )

    logger.info("Completed step resolver node.")

    return response
