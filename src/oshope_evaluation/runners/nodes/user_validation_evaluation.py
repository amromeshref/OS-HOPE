from oshope_evaluation.runners.pydantic_schemas import UserValidationEvaluation
from oshope_evaluation.config import (
    DEFAULT_LLM_PLATFORM,
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_NAME,
)
from oshope_evaluation.runners.prompts.user_validation_evaluation import (
    get_user_validation_evaluation_system_prompt,
    get_human_message_for_user_validation_evaluation,
)
from oshope.core.states.oshope_state import OSHopeState
from oshope.core.models.main import LLMModel
from oshope.utils.logger import get_logger

logger = get_logger(__name__)


def user_validation_evaluation_node(state: OSHopeState) -> UserValidationEvaluation:
    logger.info("Starting User Validation Evaluation Node")

    if state.user_validation_attempts == 0:
        logger.info("No user validation attempts were made, skipping evaluation.")
        return UserValidationEvaluation(evaluation_needed=False)

    # Get the system prompt for evaluation
    system_prompt = get_user_validation_evaluation_system_prompt()

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

    human_message = get_human_message_for_user_validation_evaluation(state)

    # Generate the evaluation result using the LLM
    response: UserValidationEvaluation = llm_model.generate_response(
        system_message=system_prompt,
        human_message=human_message,
        structured_output=UserValidationEvaluation,
    )

    response.evaluation_needed = True

    logger.info("Completed User Validation Evaluation Node")

    return response
