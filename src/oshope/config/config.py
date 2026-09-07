import os

# Voice input settings
VOICE_INPUT_ENABLED = True
AVAILABLE_VOICE_PLATFORMS = ["whisper", "google"]
DEFAULT_VOICE_PLATFORM = "google"

# RAG settings
RAG_ENABLED = True
AVAILABLE_EMBEDDING_MODELS_PLATFORMS = ["openai", "groq", "huggingface"]
DEFAULT_EMBEDDING_MODEL_PLATFORM = "huggingface"
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_NAME = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
AVAILABLE_INGESTION_PLATFORMS = ["chromadb"]
DEFAULT_INGESTION_PLATFORM = "chromadb"
AVAILABLE_RETRIEVER_PLATFORMS = ["chromadb"]
DEFAULT_RETRIEVER_PLATFORM = "chromadb"
DEFAULT_TOP_K = 3

# LLM Platforms
AVAILABLE_LLM_PLATFORMS = ["ollama", "groq"]
DEFAULT_LLM_PLATFORM = "groq"

# Ollama settings
DEFAULT_OLLAMA_MODEL_NAME = "qwen2.5:14b-instruct-q5_K_M"

# Groq settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_1 = "openai/gpt-oss-120b"
GROQ_MODEL_2 = "openai/gpt-oss-20b"
DEFAULT_GROQ_MODEL_NAME = GROQ_MODEL_1


# Node Names
QUERY_CLASSIFICATION_NODE = "query_classification_node"
QUERY_CLARIFICATION_NODE = "query_clarification_node"
PLANNING_NODE = "planning_node"
USER_VALIDATION_NODE = "user_validation_node"
EXECUTION_ORCHESTRATOR_NODE = "execution_orchestrator_node"
CODE_EXECUTION_NODE = "code_execution_node"
INFORMATION_NODE = "information_generation_node"
FINAL_RESPONSE_NODE = "final_response_node"
STEP_RESOLVER_NODE = "step_resolver_node"
CODE_ERROR_HANDLING_NODE = "code_error_handling_node"
SUMMARIZER_NODE = "summarizer_node"

# Configuration for the clarification node
CLARIFICATION_NODE_MAX_ATTEMPTS = 10

# Configuration for the command error handling node
COMMAND_ERROR_HANDLING_MAX_ATTEMPTS = 5

# Parallel Execution Settings
PARALLEL_EXECUTION_ENABLED = True
MAX_WORKERS = 4

# Debug settings
DEBUG_MODE = True
