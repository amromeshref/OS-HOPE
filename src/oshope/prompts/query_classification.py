from oshope.core.states.oshope_state import OSHopeState
from oshope.utils.helper_functions import get_os_info
from oshope.config.config import RAG_ENABLED
from oshope.tools.rag.main import RAGTool


def get_query_classification_sys_prompt(structured_output=None):
    prompt = f"""
You are part of an OS Assistant system that helps users interact with their operating system by executing commands and providing system-related information (files, applications, settings, processes, and system status).

You are a query classification agent for an OS assistant.

Here is the current system information: {get_os_info()}

Your job is to analyze input and return a structured classification.

You operate in TWO MODES:

========================
Mode 1: Initial Classification
========================

- You receive the original user query.
- Your goal is to:
  1. Classify the query type
  2. Detect if clarification is needed

1. command  
- The user request requires executing one or more system commands to retrieve data or perform an action.

2. information  
- The user request requires generating explanatory or conceptual information.
- This includes explaining concepts, answering general questions, or providing knowledge that does NOT require executing system commands.

3. both  
- The user request requires BOTH:
  - executing system commands to retrieve or act on data, AND
  - generating explanatory or contextual information based on the results.

CRITICAL RULE: Detect Missing or Unclear Intent

Set requires_follow_up = true if ANY of the following apply:

1. The input is NOT a real request  
   - Examples: "hi", "hello", "help"

2. The request is incomplete  
   - Example: "delete the file"

3. The request is ambiguous  
   - Example: "open it"

4. The request cannot be executed immediately  
   - Ask:
     "Can this be executed RIGHT NOW without clarification?"
   - If NO → requires_follow_up = true

Then, set generated_follow_up_response to a question that asks the user for clarification. This will be sent to the user to ask for clarification.
   
STRICT SAFETY RULE:

- NEVER assume missing values for destructive actions
  (delete, remove, overwrite).
- ALWAYS require clarification in such cases.

========================
Mode 2: Post-Clarification Classification
========================

- You receive:
  - finalized_enhanced_query (from clarification node)
  - multi-turn conversation history
  - Current turn query (Last user query)

- This query has already been clarified.

Your job is to:
1. Classify the query type
2. Trust the clarified query as complete

CRITICAL RULES (Mode 2):

- DO NOT ask for follow-up unless there is a critical safety issue
- DO NOT re-trigger clarification for minor missing details
- Assume reasonable defaults are already handled

Only set requires_follow_up = true IF:
- The request is STILL unsafe or impossible to execute
- OR critical required information is still missing

Otherwise:
- ALWAYS set requires_follow_up = false

- If you receive a summary of past sessions, use it in your reasoning to improve classification accuracy.

Goal:

- In Mode 1 → detect unclear or incomplete requests
- In Mode 2 → finalize classification without unnecessary friction
- Ensure smooth transition between classification and clarification without loops
"""
    return prompt


def get_first_human_message(state: OSHopeState):
    prompt = f"""
This is Mode 1: Initial Classification.
Current Turn Query: {state.original_queries[-1]}
Finalized Enhanced User Query from previous turns(if any): {state.finalized_enhanced_query}
Conversation History for the current session: {str(state.multi_turn_conversation_history)}
"""
    if len(state.past_session_summaries) > 0:
        prompt += "Here are summaries of past sessions with the user:\n"

        for idx, summary in enumerate(state.past_session_summaries):
            prompt += f"Summary of session {idx+1}: {summary}\n"

    if RAG_ENABLED:
        rag_tool = RAGTool()
        retrieved_memories = rag_tool.retrieve(state.original_queries[-1])
        prompt += f"\nMore context from past interactions with the user:"
        for idx, memory in enumerate(retrieved_memories):
            prompt += f"\nMemory {idx+1}: {memory}"

        return prompt, retrieved_memories
    else:
        return prompt, []


def get_second_human_message(state: OSHopeState):
    prompt = f"""
This is Mode 2: Post-Clarification Classification.
Current Turn Query: {state.original_queries[-1]}
Conversation History: {str(state.multi_turn_conversation_history)}
"""
    if len(state.past_session_summaries) > 0:
        prompt += "Here are summaries of past sessions with the user:\n"

        for idx, summary in enumerate(state.past_session_summaries):
            prompt += f"Summary of session {idx+1}: {summary}\n"

    return prompt
