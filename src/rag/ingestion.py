# src/rag/ingestion.py
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path
import logging
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from langchain_classic.schema import Document
from langchain_classic.text_splitter import MarkdownTextSplitter, HTMLHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
import markdown

logger = logging.getLogger(__name__)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def _read_markdown_files(folder: str) -> List[Tuple[str, str, str]]:
    files: List[Tuple[str, str, str]] = []
    p = Path(folder)
    if not p.exists():
        logger.warning("Articles folder does not exist: %s", folder)
        return files

    for md_path in p.rglob("*.md"):
        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", md_path, e)
            continue
        files.append((md_path.name, str(md_path), text))
    return files


def load_products_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not csv_path or not Path(csv_path).exists():
        logger.warning("Products CSV not found at: %s", csv_path)
        return []
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    rows = df.to_dict(orient="records")
    logger.info("Loaded %d products from CSV", len(rows))
    return rows


def ingest_corpus(articles_dir: str,products_csv_path: str = None,chunk_size_chars: int = 2000,chunk_overlap: int = 400,) -> Tuple[List[Document], List[str], List[List[str]]]:
    """
    Ingest markdown articles (and optional products CSV).
    Uses Markdown -> HTML -> HTMLHeaderTextSplitter + MarkdownTextSplitter for robust semantic chunks.
    Returns: docs, raw_texts, tokenized_for_bm25
    """
    docs: List[Document] = []
    raw_texts: List[str] = []

    md_files = _read_markdown_files(articles_dir)
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    header_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    char_splitter = MarkdownTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap)

    for filename, filepath, md_content in md_files:
        name_lower = Path(filename).stem.lower()
        if "faq" in name_lower or len(md_content.split()) < 300:
            metadata = {"doc_name": name_lower, "section_id": "full", "source": filepath}
            doc = Document(page_content=md_content, metadata=metadata)
            docs.append(doc)
            raw_texts.append(md_content)
            continue
        try:
            html = markdown.markdown(md_content)
        except Exception:
            html = "<div>" + md_content + "</div>"

        header_sections = header_splitter.split_text(html)
        section_counter = 0
        for sec in header_sections:
            chunks = char_splitter.split_text(sec.page_content)
            for i, c in enumerate(chunks):
                plain = c.page_content if hasattr(c, "page_content") else str(c)
                metadata = {
                    "doc_name": name_lower,
                    "section_id": f"sec{section_counter}_chunk{i}",
                    "source": filepath,
                    "header": sec.metadata.get("header") if hasattr(sec, "metadata") else None,
                }
                doc = Document(page_content=plain, metadata=metadata)
                docs.append(doc)
                raw_texts.append(plain)
            section_counter += 1

    logger.info("Ingested %d article document chunks", len(docs))

    if products_csv_path:
        rows = load_products_csv(products_csv_path)
        for row in rows:
            product_id = str(row.get("product_id") or row.get("product_id".upper()) or row.get("product id") or "")
            name = row.get("name") or row.get("Name") or ""
            parts: List[str] = []
            for k, v in row.items():
                if pd.isna(v):
                    continue
                parts.append(f"{k}: {v}")
            content = "\n".join(parts)
            section_id = product_id if product_id else (name.replace(" ", "_")[:40] if name else "product")
            metadata = {
                "doc_name": "products_csv",
                "section_id": section_id,
                "source": products_csv_path,
                "product_id": product_id,
                "product_name": name,
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
            raw_texts.append(content)
        logger.info("Added %d product docs from %s", len(rows), products_csv_path)
    tokenized_for_bm25 = [word_tokenize(t.lower()) for t in raw_texts]
    return docs, raw_texts, tokenized_for_bm25


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.getcwd()
    articles_dir = os.path.join(base, "data", "articles")
    products_csv = os.path.join(base, "data", "products_catalog.csv")
    docs, raw_texts, tokenized = ingest_corpus(articles_dir, products_csv)
    print(f"Ingested {len(docs)} documents. Sample metadata (first 5):")
    for d in docs[:5]:
        print(d.metadata)