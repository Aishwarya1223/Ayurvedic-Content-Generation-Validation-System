import logging
import nltk
import hashlib
from langchain_classic.schema import Document

logger = logging.getLogger(__name__)


def _normalize_doc(doc):
    if isinstance(doc, Document):
        return doc

    meta = getattr(doc, "metadata", None)
    if meta is None and isinstance(doc, dict):
        meta = doc.get("metadata", {})

    text = getattr(doc, "page_content", None)
    if text is None and isinstance(doc, dict):
        text = doc.get("document")

    if text is None:
        text = str(doc)

    return Document(page_content=text, metadata=meta or {})


def _dedupe_key(doc):
    meta = doc.metadata or {}

    if "id" in meta:
        return ("id", meta["id"])

    if "product_id" in meta:
        return ("product", meta["product_id"])

    if meta.get("source") or meta.get("section_id"):
        return (meta.get("source"), meta.get("section_id"))

    snippet = (doc.page_content or "")[:200]
    h = hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:10]
    return ("hash", h)


def hybrid_retrieve(query,vectordb,embed,bm25=None,tokenized_docs=None,tokenized_docs_meta=None,top_k_emb=6,top_k_bm25=3,metadata_filter=None,):

    q_vec = None
    if embed is None:
        logger.warning("hybrid_retrieve: embed is None — vector search disabled.")
    else:
        try:
            if hasattr(embed, "embed_query"):
                q_vec = embed.embed_query(query)
            else:
                v = embed.embed_documents([query])
                if v:
                    q_vec = v[0]
        except Exception as e:
            logger.warning("Embedding failed: %s", e)

    emb_hits = []
    if q_vec is not None:
        try:
            emb_raw = vectordb.similarity_search_by_vector(q_vec, k=top_k_emb,filter=metadata_filter)
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            emb_raw = []

        emb_hits = [_normalize_doc(d) for d in emb_raw]
    else:
        emb_hits = []


    bm25_hits = []
    if bm25 and tokenized_docs:
        try:
            words = nltk.word_tokenize(query.lower())
            scores = bm25.get_scores(words)
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k_bm25]

            for i in top_idx:
                if isinstance(tokenized_docs[i], str):
                    text = tokenized_docs[i]
                else:
                    text = " ".join(tokenized_docs[i])

                if tokenized_docs_meta and i < len(tokenized_docs_meta):
                    meta = tokenized_docs_meta[i]
                else:
                    meta = {"bm25_index": i}

                bm25_hits.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)
    merged = []
    seen = set()

    for d in emb_hits + bm25_hits:
        d = _normalize_doc(d)
        k = _dedupe_key(d)
        if k not in seen:
            seen.add(k)
            merged.append(d)

    logger.debug(
        "hybrid_retrieve: emb=%d bm25=%d total=%d",
        len(emb_hits),
        len(bm25_hits),
        len(merged),
    )

    return merged
