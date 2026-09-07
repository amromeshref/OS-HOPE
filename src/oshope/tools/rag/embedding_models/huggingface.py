from oshope.config.config import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from oshope.utils.logger import get_logger
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

logger = get_logger(__name__)


class HuggingFaceEmbeddingModel:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL

        logger.info(f"Initializing HuggingFace embedding model: {model_name}")

        self.model_name = model_name

        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def generate_embedding(self, text: str) -> list:
        logger.info(
            f"Generating embedding for text: {text[:30]}..."
        )  # Log the beginning of the text for reference
        # Tokenize input text
        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling to get sentence embedding
        sentence_embedding = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize the embedding
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        logger.info(f"Embedding generated successfully.")

        return sentence_embedding.squeeze().tolist()
