# src/tools/fact_checker_tools.py
from typing import Any, Dict, List, Optional
import json
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from langchain_core.tools import Tool
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate

from src.rag.retriever import hybrid_retrieve
from src.utils import prompts
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


CITATION_TAG_RE = re.compile(r"\[([^\[\]]+)\]")

def _extract_citation_tags(text: str) -> List[str]:
    return [m.group(1) for m in CITATION_TAG_RE.finditer(text)]


def _lexical_overlap_score(a: str, b: str) -> float:
    toks_a = [t.lower() for t in word_tokenize(a) if t.isalnum()]
    toks_b = [t.lower() for t in word_tokenize(b) if t.isalnum()]
    if not toks_a:
        return 0.0
    return len(set(toks_a) & set(toks_b)) / max(1, len(set(toks_a)))

def fact_check_draft(draft_text: str,vectordb,embed,bm25=None,tokenized_docs=None,top_k: int = 3,threshold: float = 0.25):

    sentences = sent_tokenize(draft_text)
    results = []
    matched_count = 0

    for i, s in enumerate(sentences):
        s_clean = s.strip()
        if not s_clean:
            results.append({
                "idx": i, "sentence": s, "matched": False,
                "overlap_score": 0.0,
                "citations_in_draft": [], "top_matches": []
            })
            continue

        citation_tags = _extract_citation_tags(s_clean)

        hits = hybrid_retrieve(
            query=s_clean,
            vectordb=vectordb,
            embed=embed,
            bm25=bm25,
            tokenized_docs=tokenized_docs,
            top_k_emb=top_k,
            top_k_bm25=2,
        )

        best_overlap = 0.0
        top_matches = []

        for h in hits:
            text = h.page_content or ""
            meta = h.metadata or {}
            overlap = _lexical_overlap_score(s_clean, text)
            top_matches.append({
                "doc_meta": meta,
                "snippet": text[:300],
                "overlap": overlap,
            })
            best_overlap = max(best_overlap, overlap)

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

    return {
        "per_sentence": results,
        "summary_score": summary_score,
        "corrected_draft": " ".join(corrected),
    }


def fact_check_tool_factory(vectordb, embed, bm25=None, tokenized_docs=None):
    def _tool(input_str: str) -> str:
        try:
            if isinstance(input_str, dict):
                payload = input_str
            else:
                payload = json.loads(input_str)
        except Exception:
            payload = {"draft": input_str}

        draft = payload.get("draft", "")
        fc = fact_check_draft(
            draft_text=draft,
            vectordb=vectordb,
            embed=embed,
            bm25=bm25,
            tokenized_docs=tokenized_docs,
        )
        return json.dumps(fc, ensure_ascii=False)

    return Tool(
        name="fact_check",
        description="Run deterministic hybrid-retrieval fact check.",
        func=_tool,
    )

def format_report_tool_factory(llm, name: str = "format_report"):
    """
    Returns a langchain Tool that takes either:
      - a JSON string containing {"draft": "...", "fact_check": {...}}
      - or a plain draft string
    and returns a strict JSON string:
      {"summary": "...", "grounding_score": float, "unsupported_sentences": [...], "suggested_draft": "..."}
    """

    prompt = PromptTemplate(
        input_variables=["draft", "fact_check_json"],
        template=(
            "You are a fact-checking assistant for Kerala Ayurveda.\n"
            "Your job is to check the DRAFT against FACT_CHECK_JSON (retrieved evidence).\n\n"
            "RULES:\n"
            "- Do NOT add new claims.\n"
            "- Do NOT apply tone editing.\n"
            "- Only check factual support.\n"
            "- Suggested draft must keep meaning but remove/soften unsupported claims.\n\n"
            "FACT_CHECK_JSON:\n{fact_check_json}\n\n"
            "DRAFT TO CHECK:\n{draft}\n\n"
            "Return ONLY this JSON object (no markdown, no explanation):\n"
            "{{\n"
            "  \"summary\": \"...\",\n"
            "  \"grounding_score\": 0.0,\n"
            "  \"unsupported_sentences\": [\n"
            "      {{\"idx\": 0, \"sentence\": \"...\", \"reason\": \"...\"}}\n"
            "  ],\n"
            "  \"suggested_draft\": \"...\"\n"
            "}}\n\n"
            "If invalid JSON, return exactly:\n"
            "{{\"summary\":\"invalid\",\"grounding_score\":0.0,\"unsupported_sentences\":[],\"suggested_draft\":\"\"}}"
        )
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    def _sanitize_unsupported_list(items: Any):
        if not isinstance(items, list):
            return []
        sanitized = []
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("idx")
            try:
                idx = int(idx) if idx is not None else None
            except Exception:
                idx = None
            sentence = it.get("sentence", "")
            reason = it.get("reason", "")
            sanitized.append({
                "idx": idx if idx is not None else 0,
                "sentence": sentence if isinstance(sentence, str) else str(sentence),
                "reason": reason if isinstance(reason, str) else str(reason)
            })
        return sanitized

    def _tool(input_str: str) -> str:
        payload = None
        if not input_str:
            payload = {"draft": ""}
        else:
            try:
                payload = json.loads(input_str)
                if not isinstance(payload, dict):
                    payload = {"draft": str(input_str)}
            except Exception:
                payload = {"draft": input_str}

        draft = payload.get("draft", "") or ""
        fact_check_obj = payload.get("fact_check", payload.get("fact_check_json", {}))
        try:
            fc_json = json.dumps(fact_check_obj, ensure_ascii=False)
        except Exception:
            fc_json = json.dumps({}, ensure_ascii=False)

        try:
            out = chain.run({"draft": draft, "fact_check_json": fc_json})
        except Exception as e:
            logger.exception("format_report_tool: chain.run failed: %s", e)
            fallback = {
                "summary": "chain_run_failed",
                "grounding_score": 0.0,
                "unsupported_sentences": [],
                "suggested_draft": draft
            }
            return json.dumps(fallback, ensure_ascii=False)

        logger.debug("format_report_tool RAW LLM OUTPUT: %r", out)

        out_text = (out or "").strip()
        # remove ```json ... ``` or ``` ... ```
        out_text = re.sub(r"```(?:json)?\s*", "", out_text)
        out_text = re.sub(r"\s*```$", "", out_text)

        parsed = None
        try:
            parsed = json.loads(out_text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", out_text)
            if m:
                candidate = m.group(0)
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            summary = parsed.get("summary", "")
            grounding_score = parsed.get("grounding_score", 0.0)
            unsupported_sentences = parsed.get("unsupported_sentences", [])
            suggested_draft = parsed.get("suggested_draft", draft)

            if not isinstance(summary, str):
                summary = str(summary)
            try:
                grounding_score = float(grounding_score)
                grounding_score = max(0.0, min(1.0, grounding_score))
            except Exception:
                grounding_score = 0.0

            sanitized_unsupported = _sanitize_unsupported_list(unsupported_sentences)
            if not isinstance(suggested_draft, str):
                suggested_draft = str(suggested_draft)

            safe_result = {
                "summary": summary,
                "grounding_score": grounding_score,
                "unsupported_sentences": sanitized_unsupported,
                "suggested_draft": suggested_draft
            }
            return json.dumps(safe_result, ensure_ascii=False)
        logger.warning("format_report_tool: LLM output not JSON; returning fallback. RAW: %r", out_text)
        fallback = {
            "summary": "invalid",
            "grounding_score": 0.0,
            "unsupported_sentences": [],
            "suggested_draft": ""
        }
        return json.dumps(fallback, ensure_ascii=False)

    return Tool(
        name=name,
        func=_tool,
        description="Format fact-check JSON into editor report. Input: JSON or draft string. Output: JSON string.",
        return_direct=True
    )

def default_tools(vectordb, embed, bm25=None, tokenized_docs=None, llm=None):
    return [
        fact_check_tool_factory(vectordb, embed, bm25=bm25, tokenized_docs=tokenized_docs),
        format_report_tool_factory(llm=llm),
    ]