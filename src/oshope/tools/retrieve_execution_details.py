from oshope.core.states.oshope_state import CommandExecution, OSHopeState
from oshope.utils.logger import get_logger
from langchain.tools import Tool
from typing import List

logger = get_logger(__name__)


def command_executions_to_str(executions: List[CommandExecution]) -> str:
    """
    Convert a list of CommandExecution objects into a human-readable string format for debugging or prompting purposes.
    Args:
        executions (List[CommandExecution]): The list of CommandExecution objects to convert.
    Returns:
        str: A human-readable string representation of the command executions.
    """
    if not executions:
        return "No command executions."

    lines = []
    for i, exec in enumerate(executions, start=1):
        lines.append(f"Step Index: {exec.step_index}")
        lines.append(f"Command: {exec.command}")
        lines.append(f"  Success: {'Yes' if exec.success else 'No'}")
        lines.append(f"  Output: {exec.output or 'no output'}")
        lines.append(f"  Error: {exec.error or 'no errors'}")
        lines.append("")  # empty line for separation

    return "\n".join(lines)


def retrieve_execution_details(
    state: OSHopeState,
    step_index: int,
) -> str:
    """
    Retrieve the execution details of a specific command.
    Args:
        state: OSHopeState
        step_index: The index of the step for which to retrieve execution details
    Returns:
        str: The execution details of the command.
    """
    logger.info("Starting retrieval of command execution details tool")

    try:
        command_executions = []
        if state.parallel_execution_enabled:
            for command_exe in state.planning.plan_steps[step_index].command_executions:
                command_executions.append(command_exe)
        else:
            for command_exe in state.command_executions:
                if command_exe.step_index == step_index:
                    command_executions.append(command_exe)

        command_executions_str = command_executions_to_str(command_executions)

        logger.info("Completed retrieval of command execution details tool")
        return command_executions_str
    except:
        return "No command executions found. Check the step index or try the 'retrieve_information_details'"


retrieve_execution_details_tool = Tool(
    name="RetrieveExecutionDetails",
    func=retrieve_execution_details,
    description="Retrieve the execution details of a specific command based on a query.",
)
