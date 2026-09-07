import os

# LLM Platforms
AVAILABLE_LLM_PLATFORMS = ["ollama", "groq"]
DEFAULT_LLM_PLATFORM = "groq"

# Ollama settings
DEFAULT_OLLAMA_MODEL_NAME = "llama3:8b"

# Groq settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_1 = "llama-3.3-70b-versatile"
GROQ_MODEL_2 = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL_NAME = GROQ_MODEL_1

# Dataset
DATASET_VERSION = "v1"

# Parallel execution
PARALLEL_EXECUTION_ENABLED = False

# RAG settings
RAG_ENABLED = True

# Debug mode
DEBUG_MODE = True
