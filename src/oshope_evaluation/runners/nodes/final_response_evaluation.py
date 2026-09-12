from oshope_evaluation.runners.pydantic_schemas import FinalResponseEvaluation
from oshope_evaluation.config import (
    DEFAULT_LLM_PLATFORM,
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_NAME,
)
from oshope_evaluation.runners.prompts.final_response_evaluation import (
    get_final_response_evaluation_system_prompt,
    get_human_message_for_final_response_evaluation,
)
from oshope.core.states.oshope_state import OSHopeState
from oshope.core.models.main import LLMModel
from oshope.utils.logger import get_logger

logger = get_logger(__name__)


def final_response_evaluation_node(state: OSHopeState) -> FinalResponseEvaluation:
    logger.info("Starting Final Response Evaluation Node")

    # Get the system prompt for evaluation
    system_prompt = get_final_response_evaluation_system_prompt()

    if DEFAULT_LLM_PLATFORM == "groq":
        llm_model = LLMModel(
            platform=DEFAULT_LLM_PLATFORM, model_name=DEFAULT_GROQ_MODEL_NAME
        )
    elif DEFAULT_LLM_PLATFORM == "ollama":
        llm_model = LLMModel(
            platform=DEFAULT_LLM_PLATFORM, model_name=DEFAULT_OLLAMA_MODEL_NAME
        )
    else:
        logger.error(f"Unsupported LLM platform: {DEFAULT_LLM_PLATFORM}")
        raise ValueError(f"Unsupported LLM platform: {DEFAULT_LLM_PLATFORM}")

    human_message = get_human_message_for_final_response_evaluation(state)

    # Generate the evaluation result using the LLM
    response: FinalResponseEvaluation = llm_model.generate_response(
        system_message=system_prompt,
        human_message=human_message,
        structured_output=FinalResponseEvaluation,
    )

    logger.info("Completed Final Response Evaluation Node")

    return response
