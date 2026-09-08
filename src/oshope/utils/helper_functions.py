from oshope.tools.retrieve_execution_details import retrieve_execution_details
from oshope.tools.retrieve_information_details import retrieve_information_details
from oshope.core.states.oshope_state import (
    CommandExecution,
    PlanningState,
    InformationResponse,
    InformationStep,
    CommandStep,
    Step,
    OSHopeState,
    CommandErrorHandlerState,
)
import json
from pathlib import Path
from datetime import datetime
from typing import Union, List
import os
import platform
import tempfile

PARENT_DIR = Path(__file__).parent.parent.parent.parent

DEBUG_STATE_PATH = PARENT_DIR / "logs" / "debug_state.json"
COMMAND_EXECUTIONS_JSON_FILE = PARENT_DIR / "logs" / "command_executions.json"
INFORMATION_RESPONSES_JSON_FILE = PARENT_DIR / "logs" / "information_responses.json"


def save_debug_state(
    state, title: str, path: Union[str, Path] = DEBUG_STATE_PATH
) -> None:
    """ """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create new entry
    entry = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        "state": state.model_dump(),
    }

    # Load existing data if file exists
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    # Append new state
    data.append(entry)

    # Save back
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def planning_state_to_str(state: PlanningState) -> str:
    """
    Convert a PlanningState object into a human-readable string format for debugging or prompting purposes.

    Args:
        state (PlanningState): The PlanningState object to convert.

    Returns:
        str: A human-readable string representation of the PlanningState.
    """
    lines = []

    # Fulfillment summary
    if state.fulfillment_summary:
        lines.append(f"Fulfillment Summary:\n{state.fulfillment_summary}\n")

    # Plan steps
    if state.plan_steps:
        lines.append("Plan Steps:")
        for i, step in enumerate(state.plan_steps, start=0):
            lines.append(f"{i}. Step Description: {step.description}")
            lines.append(f"   Step Type: {step.step_type}")

            # Information step
            if step.step_type == "information" and isinstance(
                step.step_details, InformationStep
            ):
                lines.append(f"   Details: {step.step_details.description}")

            # Command step
            elif step.step_type == "command" and isinstance(
                step.step_details, CommandStep
            ):
                cmd = step.step_details
                lines.append(f"   Command: {cmd.command}")
                lines.append(f"   Description: {cmd.description}")
                lines.append(
                    f"   Expected Output: {cmd.expected_output or 'no output'}"
                )
                lines.append(f"   Safety Risk: {cmd.safety_risk}")

                # Input variables
                if cmd.input_variables:
                    lines.append("   Input Variables:")
                    for var in cmd.input_variables:
                        lines.append(f"      - {var.variable_name}: {var.description}")

                # Output variables
                if cmd.output_variables:
                    lines.append("   Output Variables:")
                    for var in cmd.output_variables:
                        lines.append(f"      - {var.variable_name}: {var.description}")

            lines.append("")  # empty line between steps

    # Follow-up reasoning
    if state.requires_follow_up:
        lines.append("Requires Follow-Up: Yes")
        if state.follow_up_reasoning:
            lines.append(f"Follow-Up Reasoning: {state.follow_up_reasoning}")
    else:
        lines.append("Requires Follow-Up: No")

    return "\n".join(lines)


def plan_step_to_str(step: Step) -> str:
    """
    Convert a Step object into a human-readable string format for debugging or prompting purposes.

    Args:
        step (Step): The Step object to convert.
    Returns:
        str: A human-readable string representation of the Step.
    """
    lines = []
    lines.append(f"Step Description: {step.description}")
    lines.append(f"Step Type: {step.step_type}")

    lines.append("Step Details: ")

    # Information step
    if step.step_type == "information" and isinstance(
        step.step_details, InformationStep
    ):
        lines.append(f"Description: {step.step_details.description}")

    # Command step
    elif step.step_type == "command" and isinstance(step.step_details, CommandStep):

        cmd = step.step_details
        lines.append(f"Command: {cmd.command}")
        lines.append(f"Description: {cmd.description}")
        lines.append(f"Expected Output: {cmd.expected_output or 'no output'}")
        lines.append(f"Safety Risk: {cmd.safety_risk}")

        # Input variables
        if cmd.input_variables:
            lines.append("Input Variables:")
            for var in cmd.input_variables:
                lines.append(f"   - {var.variable_name}: {var.description}")

        # Output variables
        if cmd.output_variables:
            lines.append("Output Variables:")
            for var in cmd.output_variables:
                lines.append(f"   - {var.variable_name}: {var.description}")

        # Iteration Part
        lines.append(f"Requires Iteration: {str(step.requires_iteration)}")
        lines.append(f"Dependencies Required: {str(step.dependencies_required)}")
        lines.append(f"Dependency Step Indices: {str(step.dependency_step_indices)}")

    return "\n".join(lines)


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


def information_responses_to_str(responses: List[InformationResponse]) -> str:
    """
    Convert a list of InformationResponse objects into a human-readable string format for debugging or prompting purposes.
    Args:
        responses (List[InformationResponse]): The list of InformationResponse objects to convert.
    Returns:
        str: A human-readable string representation of the information responses.
    """
    if not responses:
        return "No information responses."

    lines = []
    for i, response in enumerate(responses, start=1):
        lines.append(f"Step Index: {response.step_index}")
        lines.append(f"  Query: {response.query or 'N/A'}")
        lines.append(f"  Answer: {response.answer or 'N/A'}")
        lines.append("")  # empty line for readability

    return "\n".join(lines)


def get_os_info() -> str:
    """
    Returns a human-readable summary of the operating system information.
    """
    lines = []

    # Basic OS info
    lines.append(f"System: {platform.system()}")
    lines.append(f"Node Name: {platform.node()}")
    lines.append(f"Release: {platform.release()}")
    lines.append(f"Version: {platform.version()}")
    lines.append(f"Machine: {platform.machine()}")
    lines.append(f"Processor: {platform.processor()}")

    # Python info
    lines.append(f"Python Version: {platform.python_version()}")

    # Current working directory
    lines.append(f"Current Directory: {os.getcwd()}")

    return "\n".join(lines)


def get_temp_dir() -> Path:
    """
    Returns the system temporary directory as a Path object.
    """
    return Path(tempfile.gettempdir())


def save_command_executions(
    executions: List[CommandExecution], filename: str = COMMAND_EXECUTIONS_JSON_FILE
) -> None:
    """
    Saves a list of CommandExecution objects to a JSON file in the temp directory.
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    # Convert to dicts
    data = [exec.dict() for exec in executions]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_command_executions(
    filename: str = COMMAND_EXECUTIONS_JSON_FILE,
) -> List[CommandExecution]:
    """
    Loads CommandExecution objects from a JSON file in the temp directory.

    Returns:
        List[CommandExecution]
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    if not file_path.exists():
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Handle empty file
            if not content:
                return []

            data = json.loads(content)

    except json.JSONDecodeError:
        # Handle corrupted / invalid JSON
        return []

    return [CommandExecution(**item) for item in data]


def save_information_responses(
    responses: List[InformationResponse],
    filename: str = INFORMATION_RESPONSES_JSON_FILE,
) -> None:
    """
    Saves a list of InformationResponse objects to a JSON file in the temp directory.
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    data = [res.dict() for res in responses]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_information_responses(
    filename: str = INFORMATION_RESPONSES_JSON_FILE,
) -> List[InformationResponse]:
    """
    Loads InformationResponse objects from a JSON file in the temp directory.

    Returns:
        List[InformationResponse]
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    if not file_path.exists():
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Handle empty file
            if not content:
                return []

            data = json.loads(content)

    except json.JSONDecodeError:
        # Handle invalid/corrupted JSON
        return []

    return [InformationResponse(**item) for item in data]


def delete_command_executions_file(
    filename: str = COMMAND_EXECUTIONS_JSON_FILE,
) -> bool:
    """
    Deletes the JSON file containing CommandExecution objects in the temp directory.

    Returns:
        bool: True if file was deleted, False if file did not exist.
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    if file_path.exists():
        file_path.unlink()
        return True
    return False


def delete_information_responses_file(
    filename: str = INFORMATION_RESPONSES_JSON_FILE,
) -> bool:
    """
    Deletes the JSON file containing InformationResponse objects in the temp directory.

    Returns:
        bool: True if file was deleted, False if file did not exist.
    """
    temp_dir = get_temp_dir()
    file_path = temp_dir / filename

    if file_path.exists():
        file_path.unlink()
        return True
    return False


def command_error_handler_state_to_str(state: CommandErrorHandlerState) -> str:
    """
    Convert a CommandErrorHandlerState object into a human-readable string.

    Args:
        state (CommandErrorHandlerState): The error handler state.

    Returns:
        str: Human-readable string representation.
    """
    lines = []

    # Recovery status
    lines.append(f"Can Recover: {'Yes' if state.can_recover else 'No'}")

    # Reasoning
    if state.recovery_reasoning:
        lines.append(f"Reasoning: {state.recovery_reasoning}")

    # Suggested command (only if recoverable)
    if state.can_recover:
        lines.append(f"Suggested Command: {state.suggested_command or 'N/A'}")
        lines.append(f"Safety Risk: {state.safety_risk}")

    return "\n".join(lines)


def retrieve_dependency_outputs(
    state: OSHopeState
) -> str:
    """
    Retrieve the outputs of the dependencies for the current step.
    Args:
        state: OSHopeState
    Returns:
        str: A string representation of the dependency outputs.
    """
    current_step_index = state.current_step_index
    current_step = state.planning.plan_steps[current_step_index]

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

def format_plan_for_user(plan: PlanningState) -> str:
    lines = []

    if plan.fulfillment_summary:
        lines.append("Here's the plan:")
        lines.append(plan.fulfillment_summary)
        lines.append("")

    for i, step in enumerate(plan.plan_steps, start=1):
        lines.append(f"Step {i}: {step.description}")

        if step.step_type == "command":
            details = step.step_details

            if isinstance(details, CommandStep):
                lines.append(f"  Command: {details.command}")
                lines.append(f"  What it does: {details.description}")
                lines.append(f"  Safety risk: {details.safety_risk}")

                if details.expected_output:
                    lines.append(f"  Expected output: {details.expected_output}")

        elif step.step_type == "information":
            details = step.step_details

            if isinstance(details, InformationStep):
                lines.append(f"  Information: {details.description}")

        lines.append("")

    lines.append("Would you like me to proceed?")

    return "\n".join(lines)