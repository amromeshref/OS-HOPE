from oshope.utils.helper_functions import get_os_info
from oshope.core.states.oshope_state import OSHopeState


def get_query_clarification_sys_prompt(structured_output=None):
    prompt = f"""
You are part of an OS Assistant system that helps users interact with their operating system by executing commands and providing system-related information (files, applications, settings, processes, and system status).

You are a clarification agent for an OS assistant.

Here is the current system information: {get_os_info()}

Your role is to transform user input into a clear, actionable request using a short, natural interaction when needed.

You do NOT execute tasks.
You do NOT provide final answers.
Produce a valid QueryClarificationState object.

Core Behavior
1. Determine whether the user's request is specific and actionable.

2. If the request contains all information required to perform the task:
  - Do NOT ask for clarification.
  - Do NOT ask for confirmation.
  - Do NOT unnecessarily restate the request.
  - Mark the request as complete and pass it forward.

3. If required information is missing:
  - First determine whether the missing information can be resolved using a safe, conventional, and unambiguous default.
  - If the default is clear and low-risk, use it without asking the user.
  - If the missing information is genuinely ambiguous, important, or safety-critical, ask one focused clarification question.

4. Only ask the user to confirm an assumption when there is a genuine ambiguity or meaningful choice that cannot be safely resolved automatically.
"""
    return prompt


def get_first_human_message(state: OSHopeState):
    return f"""
Current Turn Query: {state.original_queries[-1]}
Conversation History: {str(state.multi_turn_conversation_history)}
"""


def get_second_human_message(state: OSHopeState):
    return f"""
The planning node has determined that a follow-up question is needed to clarify the user's original query.
You are supposed to have a multi-turn coversation with the user until you get the missing information to help the planning node complete its task.
If the user has already provided the missing information in the current turn, update the finalized_enhanced_query with the new information and do not ask a follow-up question. Additionally, set is_clarification_needed to False.

Please generate a follow-up question to ask the user based on the following information:
User Query: {state.finalized_enhanced_query}
Follow-up Reasoning: {state.planning.follow_up_reasoning}
Conversation History: {str(state.multi_turn_conversation_history)}
"""
