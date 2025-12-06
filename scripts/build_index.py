# build_index.py
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.rag.ingestion import ingest_corpus
from src.rag.rag_langchain import build_vectorstore
from src.rag.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    articles_dir = "./data/articles"
    products_csv = "./data/products_catalog.csv"
    docs, raw_texts, tokenized = ingest_corpus(articles_dir, products_csv)
    print(docs[0].metadata)
    vectordb, embed = build_vectorstore(docs)
    print(embed)
    logger.info("build_vectorstore -> vectordb=%s, embed=%s", type(vectordb), type(embed))
    if embed is None:
        raise RuntimeError("build_vectorstore returned embed=None — check logs for model failures")

    hits = hybrid_retrieve("What are contraindications for Ashwagandha?", 
                           vectordb, 
                           embed, 
                           bm25=None, 
                           tokenized_docs=tokenized,)
    if embed is None:
        print("ERROR: No embedding model — vector search cannot run.")
        import sys
        sys.exit(1)
    print("Top hit meta:", [h.metadata for h in hits[:3]])
