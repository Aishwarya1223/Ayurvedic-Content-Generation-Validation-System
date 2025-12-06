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


CITATION_TAG_RE = re.compile(r"\[([^\[\]#]+#[^\[\]]+)\]")

def _extract_citation_tags(text: str) -> List[str]:
    return CITATION_TAG_RE.findall(text)

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

def format_report_tool_factory(llm, name="format_report"):
    tone_fragment = getattr(prompts, "TONE_EDITOR_PROMPT", "")
    tone_fragment_escaped = tone_fragment.replace("{", "{{").replace("}", "}}")

    prompt = PromptTemplate(
    input_variables=["draft", "fact_check_json"],
    template=(
        "You are a Kerala Ayurveda editor. You must strictly use ONLY the information in FACT_CHECK_JSON.\n"
        "If any part of the draft is unsupported, fix or soften it.\n"
        f"{prompts.TONE_EDITOR_PROMPT.replace('{','{{').replace('}','}}')}\n\n"
        "FACT_CHECK_JSON:\n{fact_check_json}\n\n"
        "DRAFT:\n{draft}\n\n"
        "IMPORTANT: Return ONLY a single JSON object and nothing else. No explanation, no code fences, no markdown.\n"
        "The object MUST have these keys:\n"
        "- summary (string)\n"
        "- grounding_score (number between 0.0 and 1.0)\n"
        "- unsupported_sentences (array of objects: {{\"idx\":<int>,\"sentence\":\"...\",\"reason\":\"...\"}})\n"
        "- suggested_draft (string)\n\n"
        "If you cannot produce valid JSON, return exactly this JSON (no extra text):\n"
        '{{"summary":"invalid","grounding_score":0.0,"unsupported_sentences":[],"suggested_draft":""}}'
        )
    )


    chain = LLMChain(llm=llm, prompt=prompt)

    def _tool(input_str: str) -> str:
        """
        Always return a JSON *string*. This function is defensive:
        - logs raw LLM output
        - attempts to parse JSON
        - falls back to extracting first {...} block
        - final fallback returns the explicit invalid JSON string
        """
        try:
            payload = json.loads(input_str)
        except Exception:
            try:
                payload = {"draft": input_str}
            except Exception:
                payload = {"draft": input_str}

        draft = payload.get("draft", "")
        fc_json = json.dumps(payload.get("fact_check", {}), ensure_ascii=False)

        out = None
        try:
            out = chain.run({"draft": draft, "fact_check_json": fc_json})
        except Exception as e:
            logger.exception("format_report_tool: chain.run failed: %s", e)
            return json.dumps({
                "summary": "chain_run_failed",
                "grounding_score": 0.0,
                "unsupported_sentences": [],
                "suggested_draft": draft
            }, ensure_ascii=False)
        logger.error("format_report_tool RAW LLM OUTPUT: %r", out)

        parsed = None
        if isinstance(out, dict):
            parsed = out
        else:
            out_text = (out or "").strip()
            try:
                parsed = json.loads(out_text)
            except Exception:
                import re
                m = re.search(r"\{[\s\S]*\}", out_text)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None

        if isinstance(parsed, dict):
            summary = parsed.get("summary", "")
            grounding_score = parsed.get("grounding_score", 0.0)
            unsupported_sentences = parsed.get("unsupported_sentences", [])
            suggested_draft = parsed.get("suggested_draft", draft)

            safe_result = {
                "summary": summary if isinstance(summary, str) else str(summary),
                "grounding_score": float(grounding_score) if (isinstance(grounding_score, (int, float)) and grounding_score >= 0.0) else 0.0,
                "unsupported_sentences": unsupported_sentences if isinstance(unsupported_sentences, list) else [],
                "suggested_draft": suggested_draft if isinstance(suggested_draft, str) else str(suggested_draft)
            }
            return json.dumps(safe_result, ensure_ascii=False)

        fallback = {
            "summary": "invalid",
            "grounding_score": 0.0,
            "unsupported_sentences": [],
            "suggested_draft": ""
        }
        logger.warning("format_report_tool: LLM output not JSON; returning fallback.")
        return json.dumps(fallback, ensure_ascii=False)

    return Tool(name=name, description="Format fact-check JSON into editor report.", func=_tool)


def default_tools(vectordb, embed, bm25=None, tokenized_docs=None, llm=None):
    return [
        fact_check_tool_factory(vectordb, embed, bm25=bm25, tokenized_docs=tokenized_docs),
        format_report_tool_factory(llm=llm),
    ]