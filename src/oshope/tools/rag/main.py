from oshope.tools.rag.chromadb.rag_service import RAGService
from oshope.config.config import (
    AVAILABLE_INGESTION_PLATFORMS,
    AVAILABLE_RETRIEVER_PLATFORMS,
    DEFAULT_INGESTION_PLATFORM,
    DEFAULT_RETRIEVER_PLATFORM,
)
from oshope.utils.logger import get_logger

logger = get_logger(__name__)


class RAGTool:
    def __init__(self, ingestion_platform: str = None, retriever_platform: str = None):
        if ingestion_platform is None:
            ingestion_platform = DEFAULT_INGESTION_PLATFORM
        if retriever_platform is None:
            retriever_platform = DEFAULT_RETRIEVER_PLATFORM

        if ingestion_platform not in AVAILABLE_INGESTION_PLATFORMS:
            logger.error(
                f"Ingestion platform '{ingestion_platform}' is not supported. Available options: {AVAILABLE_INGESTION_PLATFORMS}"
            )
            raise ValueError(
                f"Ingestion platform '{ingestion_platform}' is not supported. Available options: {AVAILABLE_INGESTION_PLATFORMS}"
            )
        if retriever_platform not in AVAILABLE_RETRIEVER_PLATFORMS:
            logger.error(
                f"Retriever platform '{retriever_platform}' is not supported. Available options: {AVAILABLE_RETRIEVER_PLATFORMS}"
            )
            raise ValueError(
                f"Retriever platform '{retriever_platform}' is not supported. Available options: {AVAILABLE_RETRIEVER_PLATFORMS}"
            )

        logger.info(
            f"Initializing RAGTool with ingestion platform: {ingestion_platform} and retriever platform: {retriever_platform}"
        )

        self.ingestion_platform = ingestion_platform
        self.retriever_platform = retriever_platform

        if (
            self.ingestion_platform == "chromadb"
            or self.retriever_platform == "chromadb"
        ):
            self.rag_service = RAGService()

    def add_memories(self, session_id: int, summaries: list[str]):
        if self.ingestion_platform == "chromadb":
            self.rag_service.add_memories(session_id, summaries)

    def retrieve(self, query: str, top_k: int = None) -> list[str]:
        if self.retriever_platform == "chromadb":
            return self.rag_service.retrieve(query, top_k)
