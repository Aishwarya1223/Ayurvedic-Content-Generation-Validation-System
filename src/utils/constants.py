from pathlib import Path
import os
# Repository root (computed automatically)
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

# Data directories
DATA_DIR = ROOT_DIR / "data"
ARTICLES_DIR = DATA_DIR / "articles"
PRODUCTS_CSV = DATA_DIR / "products_catalog.csv"

# Chroma vectorstore persistence
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"


CHUNK_SIZE = 4500
CHUNK_OVERLAP = 900

# Treat files containing "faq" in their name OR < 300 words as unchunked
SHORT_DOC_WORD_THRESHOLD = 300


TOP_K_EMBEDDINGS = 6
TOP_K_BM25 = 3
BM25_MIN_TOKEN_LENGTH = 1

DEFAULT_LOG_LEVEL = "INFO"

# Toggle verbose debug logs for ingestion/retrieval
DEBUG_INGESTION = False
DEBUG_RETRIEVAL = False


PRODUCT_ID_COL = "product_id"
PRODUCT_NAME_COL = "name"
PRODUCT_CONCERNS_COL = "target_concerns"
PRODUCT_HERBS_COL = "key_herbs"
PRODUCT_CATEGORY_COL = "category"


APP_NAME = "Kerala Ayurveda AI"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Internal RAG + Agentic Workflow System for Ayurveda Content Generation"


DEFAULT_SAFETY_NOTE = (
    "This content is for informational purposes and not a substitute for "
    "medical advice. Please consult a qualified healthcare provider for "
    "guidance tailored to your situation."
)

# Forbidden claims (strict brand rules)
FORBIDDEN_CLAIMS = [
    "cure",
    "guaranteed results",
    "miracle remedy",
    "treats disease",
    "replaces medication",
]

# Required phrases to maintain brand tone
PREFERRED_TONE_PHRASES = [
    "traditionally used to support",
    "may help maintain",
    "gentle support for",
    "helps you feel more",
]


ENABLE_FACTCHECK_DEBUG = False

DISABLE_OPENAI = False


__all__ = [
    "ROOT_DIR", "DATA_DIR", "ARTICLES_DIR", "PRODUCTS_CSV",
    "CHROMA_DB_DIR",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "SHORT_DOC_WORD_THRESHOLD",
    "TOP_K_EMBEDDINGS", "TOP_K_BM25",
    "DEFAULT_LOG_LEVEL",
    "PRODUCT_ID_COL", "PRODUCT_NAME_COL", "PRODUCT_CONCERNS_COL",
    "PRODUCT_HERBS_COL", "PRODUCT_CATEGORY_COL",
    "APP_NAME", "APP_VERSION", "APP_DESCRIPTION",
    "DEFAULT_SAFETY_NOTE", "FORBIDDEN_CLAIMS", "PREFERRED_TONE_PHRASES",
    "ENABLE_FACTCHECK_DEBUG", "DISABLE_OPENAI",
]