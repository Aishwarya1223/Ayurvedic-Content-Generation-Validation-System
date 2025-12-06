# src/rag/fact_check.py
from typing import List, Dict, Any
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from langchain_classic.schema import Document
from src.rag.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


CITATION_TAG_RE = re.compile(r"\[([^\[\]#]+#[^\[\]]+)\]")


def _extract_citation_tags(text: str) -> List[str]:
    return CITATION_TAG_RE.findall(text)


def _lexical_overlap_score(a: str, b: str) -> float:
    toks_a = [t.lower() for t in word_tokenize(a) if t.isalnum()]
    toks_b = [t.lower() for t in word_tokenize(b) if t.isalnum()]
    if not toks_a:
        return 0.0
    set_a = set(toks_a)
    set_b = set(toks_b)
    return len(set_a & set_b) / max(1, len(set_a))


def fact_check_draft(draft_text: str,vectordb,embed,bm25=None,tokenized_docs=None,top_k: int = 3,threshold: float = 0.25) -> Dict[str, Any]:

    sentences = sent_tokenize(draft_text)
    results = []
    matched_count = 0

    for i, s in enumerate(sentences):
        sentence = s.strip()
        if not sentence:
            results.append({
                "idx": i,
                "sentence": s,
                "matched": False,
                "overlap_score": 0.0,
                "citations_in_draft": [],
                "top_matches": []
            })
            continue

        citation_tags = _extract_citation_tags(sentence)

        hits = hybrid_retrieve(
            query=sentence,
            vectordb=vectordb,
            embed=embed,
            bm25=bm25,
            tokenized_docs=tokenized_docs,
            top_k_emb=top_k,
            top_k_bm25=2,
        )

        top_matches = []
        best_overlap = 0.0

        for h in hits:
            text = h.page_content or ""
            meta = h.metadata or {}

            overlap = _lexical_overlap_score(sentence, text)
            top_matches.append({
                "doc_meta": meta,
                "snippet": text[:300],
                "overlap": overlap,
            })

            if overlap > best_overlap:
                best_overlap = overlap

        matched = best_overlap >= threshold

        if matched:
            matched_count += 1

        results.append({
            "idx": i,
            "sentence": s,
            "matched": matched,
            "overlap_score": best_overlap,
            "citations_in_draft": citation_tags,
            "top_matches": top_matches,
        })

    summary_score = matched_count / max(1, len(sentences))

    corrected = []
    for r in results:
        if r["matched"]:
            corrected.append(r["sentence"])
        else:
            corrected.append(r["sentence"] + " <!-- UNSUPPORTED -->")

    corrected_draft = " ".join(corrected)

    return {
        "per_sentence": results,
        "summary_score": summary_score,
        "corrected_draft": corrected_draft,
    }
