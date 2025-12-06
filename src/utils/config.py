# src/utils/config.py
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    HF_TOKEN: Optional[str] = Field(None, env="HF_TOKEN")

    CHROMA_DB_DIR: str = Field("./vectorstore/chroma", env="CHROMA_DB_DIR")

    EMBED_MODEL_NAME: Optional[str] = Field(None, env="EMBED_MODEL_NAME")
    EMBED_MODEL_CANDIDATES: List[str] = Field(
        default_factory=lambda: [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        ],
        env="EMBED_MODEL_CANDIDATES",
    )

    CHAT_MODEL_NAME: str = Field("gpt-4o", env="CHAT_MODEL_NAME")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

settings = Settings()
if settings.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
if settings.HF_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HF_TOKEN