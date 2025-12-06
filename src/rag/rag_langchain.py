#src/rag/rag_langchain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import logging
from src.utils.config import settings
from pathlib import Path
from builtins import RuntimeError

logger = logging.getLogger(__name__)
OPENAI_KEY = settings.OPENAI_API_KEY

print("Loaded settings:", settings.model_dump())

def build_vectorstore(docs, embedding_model_name="sentence-transformers/all-mpnet-base-v2", persist_directory=None):
    if not docs:
        raise ValueError("build_vectorstore requires a non-empty docs list")

    persist_directory = persist_directory or settings.CHROMA_DB_DIR
    p = Path(persist_directory)

    if p.exists() and any(p.iterdir()):
        logger.info("Existing Chroma DB found. Loading it.")

        try:
            embed = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            db = Chroma(
                persist_directory=str(p),
                embedding_function=embed
            )
            logger.info("Loaded existing DB with OpenAIEmbeddings.")
            return db, embed
        except Exception as e:
            logger.warning(f"OpenAIEmbeddings failed: {e}")

        try:
            embed = HuggingFaceEmbeddings(model_name=embedding_model_name)
            db = Chroma(
                persist_directory=str(p),
                embedding_function=embed
            )
            logger.info("Loaded existing DB with HuggingFaceEmbeddings.")
            return db, embed
        except Exception as e:
            raise RuntimeError(f"Failed to load existing DB with any embedding model: {e}")
    logger.info("No existing DB found. Building new vectorstore.")

    try:
        embed = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        db = Chroma.from_documents(
            documents=docs,
            embedding_function=embed,
            persist_directory=str(p)
        )
        logger.info("Built new DB using OpenAIEmbeddings.")
        return db, embed
    except Exception as e:
        logger.warning(f"OpenAIEmbeddings failed to build DB: {e}")

    try:
        embed = HuggingFaceEmbeddings(model_name=embedding_model_name)
        db = Chroma.from_documents(
            documents=docs,
            embedding_function=embed,
            persist_directory=str(p)
        )
        logger.info("Built new DB using HuggingFaceEmbeddings.")
        return db, embed
    except Exception as e:
        raise RuntimeError(f"Failed to build vectorstore with {embedding_model_name}: {e}")
