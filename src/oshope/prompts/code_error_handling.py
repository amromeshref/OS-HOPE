from oshope.core.states.oshope_state import OSHopeState
from oshope.utils.helper_functions import get_os_info
from oshope.utils.helper_functions import (
    command_executions_to_str,
    retrieve_dependency_outputs,
    command_error_handler_state_to_str,
)


def get_code_error_handling_sys_prompt(structured_output=None):
    prompt = f"""
You are part of an OS Assistant system that helps users interact with their operating system.

You are a Command Recovery Agent.

Your responsibility is to analyze a FAILED command execution and attempt to recover by generating a new command that correctly fulfills the intended step.

You operate in TWO MODES. The mode will be explicitly provided in the prompt.

You MUST produce a valid CommandErrorHandlerState object.

Here is the current system information: {get_os_info()}

You MUST choose ONE outcome:

- Generate a corrected command (recovery success)
- OR determine that recovery is not possible

--------------------------------------------------

MODES OF OPERATION:

MODE 1: INITIAL RECOVERY
- This is the FIRST recovery attempt
- You are analyzing the original failed command

MODE 2: RETRY RECOVERY
- A previous recovery attempt was made
- The previously suggested command ALSO failed
- You will receive the previous recovery output

IMPORTANT (Mode 2):
- You MUST NOT repeat the same mistake
- You MUST analyze why your previous fix failed
- You MUST attempt a DIFFERENT recovery strategy
- If no alternative strategy exists → DO NOT attempt another fix

--------------------------------------------------

INPUT YOU RECEIVE:

1. Step description:
   - Description of the intended command step

2. Failed command execution:
   - command
   - output
   - error

3. (Optional):
   - dependency outputs (if the step depends on previous steps)
   - resolved variables (if placeholders were used)

4. (ONLY in Mode 2):
   - Previous recovery output:
     - previously suggested command
     - previous reasoning

--------------------------------------------------

YOUR GOAL:

Recover the INTENT of the step and generate a working command.

--------------------------------------------------

STEP 1: Understand the Intent

- Carefully read the step description
- Identify what the command was supposed to do

--------------------------------------------------

STEP 2: Diagnose Failure

- Analyze why the command failed
- Possible reasons include:
  - syntax issue
  - wrong path
  - missing file
  - empty input
  - bad variable resolution
  - invalid dependency output

--------------------------------------------------

STEP 3: Mode-Specific Reasoning

IF Mode 1:
- Proceed normally with diagnosis and recovery

IF Mode 2:
- Analyze:
  - Why the PREVIOUS fix failed
  - Whether the issue is:
    - still the same root cause
    - a deeper issue (dependencies, missing data, etc.)

- You MUST:
  - Avoid repeating the same fix
  - Avoid minor variations of the same command

- If the previous fix failed due to:
  - missing data
  - invalid dependencies
  - incorrect assumptions

→ You MUST STOP and mark as not recoverable (set can_recover = False)

--------------------------------------------------

STEP 4: Use Available Context

You MAY use:
- Step description
- Dependency outputs
- Resolved variables
- Command outputs

You MUST NOT:
- Invent missing data
- Assume file existence
- Guess unknown paths

--------------------------------------------------

STEP 5: Attempt Recovery

Set can_recover = True ONLY IF:
- You can construct a valid command
- The command fulfills the step intent
- The command relies ONLY on available context
- (Mode 2) The new command is meaningfully different from the previous failed fix

--------------------------------------------------

STEP 6: Fail Safely

Set can_recover = False IF:
- Required data is missing
- Dependencies are empty or invalid
- The intent cannot be fulfilled with available information
- The fix would require guessing
- (Mode 2) No new valid recovery strategy exists

--------------------------------------------------

STEP 7: Reasoning

Explain clearly in recovery_reasoning:

- Why the command failed
- (Mode 2) Why the previous fix failed
- How the new command fixes the issue (if applicable)
- What data was used

--------------------------------------------------

CRITICAL RULES:

- NEVER hallucinate values
- NEVER assume missing files or paths
- NEVER guess user intent beyond the step description
- ONLY use provided context
- NEVER repeat the same failed fix in Mode 2
- If uncertain → DO NOT FIX, set can_recover = False
"""
    return prompt


def retrieve_previous_recovery_outputs(
    state: OSHopeState,
    step_index: int = None,
) -> str:
    """
    Retrieve the previous recovery outputs for Mode 2 operation.
    """
    outputs = []
    if state.parallel_execution_enabled:
        for i in range(
            1, state.planning.plan_steps[step_index].num_error_executions + 1
        ):
            if len(state.planning.plan_steps[step_index].command_error_handlers) >= i:
                handler_state = state.planning.plan_steps[
                    step_index
                ].command_error_handlers[-i]
                outputs.append(command_error_handler_state_to_str(handler_state))
        return "\n".join(outputs)

    else:

        for i in range(1, state.num_error_executions + 1):
            if len(state.command_error_handlers) >= i:
                handler_state = state.command_error_handlers[-i]
                outputs.append(command_error_handler_state_to_str(handler_state))
        return "\n".join(outputs)


def get_first_human_message(
    state: OSHopeState,
    step_index: int = None,
):
    """
    Get the human message for the first turn of the code error handling node.
    """
    if state.parallel_execution_enabled:
        failed_command_execution = state.planning.plan_steps[
            step_index
        ].command_executions[-1]
    else:
        failed_command_execution = state.command_executions[-1]

    if state.parallel_execution_enabled:
        prompt = f"""
This is MODE 1: INITIAL RECOVERY. This is the first recovery attempt for a failed command execution.
Step description: {state.planning.plan_steps[step_index].step_details.description}
Failed command execution: {command_executions_to_str([failed_command_execution])}
    """
    else:
        prompt = f"""
This is MODE 1: INITIAL RECOVERY. This is the first recovery attempt for a failed command execution.
Step description: {state.planning.plan_steps[state.current_step_index].step_details.description}
Failed command execution: {command_executions_to_str([failed_command_execution])}
    """

    if state.planning.plan_steps[state.current_step_index].dependencies_required:
        prompt += f"\nDependency outputs: {retrieve_dependency_outputs(state)}"

    return prompt


def get_second_human_message(
    state: OSHopeState,
    step_index: int = None,
):
    """
      Get the human message for the second turn of the code error handling node (Mode 2
    operation).
    """
    if state.parallel_execution_enabled:
        failed_command_execution = state.planning.plan_steps[
            step_index
        ].command_executions[-1]
    else:
        failed_command_execution = state.command_executions[-1]

    if state.parallel_execution_enabled:
        prompt = f"""
This is mode 2: RETRY RECOVERY. A previous recovery attempts was made but the last suggested command also failed.
Step description: {state.planning.plan_steps[step_index].step_details.description}
Failed command execution: {command_executions_to_str([failed_command_execution])}
"""
    else:
        prompt = f"""
This is mode 2: RETRY RECOVERY. A previous recovery attempts was made but the last suggested command also failed.
Step description: {state.planning.plan_steps[state.current_step_index].step_details.description}
Failed command execution: {command_executions_to_str([failed_command_execution])}
"""

    if state.planning.plan_steps[state.current_step_index].dependencies_required:
        prompt += f"\nDependency outputs: {retrieve_dependency_outputs(state=state)}"

    prompt += f"""
Previous recovery outputs (from most recent to oldest): 
{retrieve_previous_recovery_outputs(state=state, step_index=step_index)}
"""

    return prompt
