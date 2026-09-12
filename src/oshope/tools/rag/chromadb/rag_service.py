from oshope.tools.rag.embedding_models.main import EmbeddingModelFactory
from oshope.config.config import DEFAULT_TOP_K, DEFAULT_EMBEDDING_MODEL_NAME
import chromadb
import os

PARENT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        )
    )
)

if DEFAULT_EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
    PRESIST_DIRECTORY = PARENT_DIR + "/database/chromadb/all-MiniLM-L6-v2-Model"


class RAGService:
    def __init__(self, persist_directory=PRESIST_DIRECTORY):

        os.makedirs(persist_directory, exist_ok=True)

        # embedding model
        self.embedding_model_factory = EmbeddingModelFactory()
        self.embedding_model = self.embedding_model_factory.embedding_model

        # persistent chroma client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # collection
        self.collection = self.client.get_or_create_collection(name="oshope_memory")

    def add_memories(self, session_id: int, summaries: list[str]):
        if not summaries:
            return

        embeddings = []
        for summary in summaries:
            embedding = self.embedding_model.generate_embedding(summary)
            embeddings.append(embedding)

        self.collection.add(
            documents=summaries,
            embeddings=embeddings,
            metadatas=[{"session_id": session_id} for _ in summaries],
            ids=[f"{session_id}_{i}" for i in range(len(summaries))],
        )

    def retrieve(self, query: str, top_k: int = None) -> list[str]:
        if top_k is None:
            top_k = DEFAULT_TOP_K

        query_embedding = self.embedding_model.generate_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results["documents"][0] if results["documents"] else []
