from oshope_evaluation.runners.pydantic_schemas import InformationGenerationEvaluation
from oshope_evaluation.config import (
    DEFAULT_LLM_PLATFORM,
    DEFAULT_GROQ_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_NAME,
)
from oshope_evaluation.runners.prompts.information_generation_evaluation import (
    get_information_generation_evaluation_system_prompt,
    get_human_message_for_information_generation_evaluation,
)
from oshope.core.states.oshope_state import InformationResponse
from oshope.core.models.main import LLMModel
from oshope.utils.logger import get_logger
from typing import List

logger = get_logger(__name__)


def information_generation_evaluation_node(
    information_responses: List[InformationResponse],
) -> List[InformationGenerationEvaluation]:
    logger.info("Starting Information Generation Evaluation Node")

    if len(information_responses) == 0:
        logger.info(
            "No information responses to evaluate. Skipping Information Generation Evaluation Node."
        )
        return [InformationGenerationEvaluation(evaluation_needed=False)]

    # Get the system prompt for evaluation
    system_prompt = get_information_generation_evaluation_system_prompt()

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

    responses = []

    for info_response in information_responses:
        logger.info(f"Evaluating Information Response: {info_response.query}")

        human_message = get_human_message_for_information_generation_evaluation(
            info_response
        )

        # Generate the evaluation result using the LLM
        response: InformationGenerationEvaluation = llm_model.generate_response(
            system_message=system_prompt,
            human_message=human_message,
            structured_output=InformationGenerationEvaluation,
        )
        response.evaluation_needed = True
        responses.append(response)

    logger.info("Completed Information Generation Evaluation Node")

    return responses
