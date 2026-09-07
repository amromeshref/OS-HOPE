from oshope.core.states.oshope_state import OSHopeState, SummarizerState
from oshope.config.config import PARALLEL_EXECUTION_ENABLED
from oshope.core.graphs.execution.parallel.state_manager import update_state
from oshope.prompts.summarizer import get_summarizer_sys_prompt, get_human_message
from oshope.utils.logger import get_logger
from oshope.core.models.main import LLMModel
from datetime import datetime
from typing import List
import json
import os

logger = get_logger(__name__)

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))))
print("Parent directory:", PARENT_DIR)


def save_session_memory(
    session_id: int,
    summaries: List[str],
    file_path: str = PARENT_DIR + "/memory/session_memory.jsonl",
) -> None:
    """
    Saves summarizer output into a JSONL memory file for RAG.
    Each line = one memory entry.
    """

    with open(file_path, "a", encoding="utf-8") as f:
        for text in summaries:
            record = {
                "session_id": session_id,
                "text": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_next_session_id(file_path = PARENT_DIR + "/memory/session_id.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("1")
        return 1

    with open(file_path, "r") as f:
        current_id = int(f.read().strip())

    new_id = current_id + 1

    with open(file_path, "w") as f:
        f.write(str(new_id))

    return new_id

def summarizer_node(state: OSHopeState) -> OSHopeState:
    """
    Node responsible for summarizing the current state of the assistant's memory and reasoning.
    """
    logger.info("Starting summarizer node.")

    sys_prompt: str = get_summarizer_sys_prompt()
    human_message: str = get_human_message(state)
    llm_model = LLMModel()

    response: SummarizerState = llm_model.generate_response(
        system_message=sys_prompt,
        human_message=human_message,
        structured_output=SummarizerState,
    )

    if not PARALLEL_EXECUTION_ENABLED:
        state.memory_extraction = response
        state.past_session_summaries.append(response.session_summary)
    else:
        update_state(state=state, response=response)
    
    # Save the extracted memory into a JSONL file for RAG
    save_session_memory(
        session_id=get_next_session_id(),
        summaries=response.summary_for_rag,
    )

    logger.info("Completed summarizer node.")
    return state