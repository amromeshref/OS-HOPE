from oshope.config.config import (
    AVAILABLE_EMBEDDING_MODELS_PLATFORMS,
    DEFAULT_EMBEDDING_MODEL_PLATFORM,
)
from oshope.tools.rag.embedding_models.huggingface import HuggingFaceEmbeddingModel
from oshope.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModelFactory:
    def __init__(self, platform: str = None, model_name: str = None):
        if platform is None:
            platform = DEFAULT_EMBEDDING_MODEL_PLATFORM
        self.platform = platform

        if self.platform == "huggingface":
            self.embedding_model = HuggingFaceEmbeddingModel(model_name)
        else:
            logger.error(
                f"Unsupported embedding model platform: {self.platform}. Available platforms are: {AVAILABLE_EMBEDDING_MODELS_PLATFORMS}"
            )
            raise ValueError(
                f"Unsupported embedding model platform: {self.platform}. Available platforms are: {AVAILABLE_EMBEDDING_MODELS_PLATFORMS}"
            )
