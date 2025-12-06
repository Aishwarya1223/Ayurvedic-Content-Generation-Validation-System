import logging
import json
import time
from pathlib import Path

from rank_bm25 import BM25Okapi
from src.rag.ingestion import ingest_corpus
from src.rag.rag_langchain import build_vectorstore
from src.utils.config import settings
from src.utils.constants import ARTICLES_DIR, PRODUCTS_CSV
from src.agents.outline_agent import OutlineAgent
from src.agents.writer_agent import WriterAgent
from src.agents.fact_checker_agent import FactCheckerAgent
from src.agents.tone_editor import ToneEditor
from scripts.evaluate import evaluate_and_persist

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def safe_load(obj):
    """If obj is a JSON string, parse it; otherwise return obj."""
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            try:
                return json.loads(obj.replace("'", '"'))
            except Exception:
                return obj
    return obj


def extract_text_field(maybe):
    """Return a best-effort string from various return shapes."""
    if maybe is None:
        return ""
    if isinstance(maybe, str):
        return maybe
    if isinstance(maybe, dict):
        for k in ("final", "edited", "draft", "text", "suggested_draft"):
            v = maybe.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # try JSON-strings inside values
        for v in maybe.values():
            if isinstance(v, str) and v.strip():
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        for k in ("edited", "draft", "text"):
                            pv = parsed.get(k)
                            if isinstance(pv, str) and pv.strip():
                                return pv
                except Exception:
                    return v
        return json.dumps(maybe, ensure_ascii=False)
    if hasattr(maybe, "text"):
        return str(getattr(maybe, "text") or "")
    return str(maybe)


def norm_to_str(out, fallback=""):
    """Normalize various tool outputs to a final text string."""
    o = safe_load(out)
    if isinstance(o, dict):
        for k in ("edited", "final", "draft", "text", "suggested_draft"):
            v = o.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # if dict but no text keys, return JSON dump
        return json.dumps(o, ensure_ascii=False)
    return str(o) if o is not None else fallback


def main():
    log.info("Ingesting articles + product data...")
    docs, raw_texts, tokenized_docs = ingest_corpus(str(ARTICLES_DIR), products_csv_path=str(PRODUCTS_CSV))
    log.info("Ingested %d chunks", len(docs))

    log.info("Building/loading vectorstore at: %s", settings.CHROMA_DB_DIR)
    vectordb, embed = build_vectorstore(docs, embedding_model_name=getattr(settings, "EMBED_MODEL_NAME", None))
    if embed is None:
        log.critical("No embedding object returned from build_vectorstore. Aborting.")
        raise SystemExit(1)
    log.info("Embedding object: %s", type(embed))

    bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None
    brief = "Write a short accessible article about Triphala and digestion support."

    # 1) Outline
    outline_agent = OutlineAgent()
    outline_resp = safe_load(outline_agent.run(brief))
    outline_text = (outline_resp or {}).get("outline") if isinstance(outline_resp, dict) else str(outline_resp or "")
    log.info("Outline produced (first 200 chars): %s", (outline_text or "")[:200])

    # 2) Writer — pass outline + small slice of raw_texts as context
    writer_agent = WriterAgent()
    outline_text = outline_resp.get("outline") if isinstance(outline_resp, dict) else str(outline_resp)

    writer_out = safe_load(writer_agent.run(brief=brief, context=outline_text))
    draft = (writer_out.get("draft") or writer_out.get("text")) if isinstance(writer_out, dict) else str(writer_out or "")
    log.info("Draft length: %d", len(draft or ""))

    # 3) Fact-check
    fc_agent = FactCheckerAgent()
    fc_resp = fc_agent.run(draft_text=draft, vectordb=vectordb, embed=embed, bm25=bm25, tokenized_docs=tokenized_docs)

    # grounding extraction (defensive)
    grounding = None
    if isinstance(fc_resp, dict):
        grounding = (
            fc_resp.get("fact_check", {}) .get("summary_score")
            or fc_resp.get("fact_check", {}) .get("grounding_score")
            or fc_resp.get("grounding_score")
        )
    log.info("Fact-check grounding score: %s", grounding)

    draft_for_tone = (fc_resp or {}).get("suggested_draft") or (fc_resp or {}).get("fact_check", {}).get("corrected_draft") or draft

    # 4) Tone edit
    tone_agent = ToneEditor()
    tone_out = tone_agent.run(draft_for_tone)

    # normalize tone_out if it's a wrapped JSON-in-text
    tone_out = safe_load(tone_out)
    final = norm_to_str(tone_out, fallback=draft_for_tone)
    tone_notes = (tone_out.get("notes") if isinstance(tone_out, dict) else {}) or {}

    result = {
        "brief": brief,
        "outline": outline_text,
        "writer_draft": draft,
        "used_for_tone": draft_for_tone,
        "fact_check": fc_resp,
        "suggested_draft": (fc_resp or {}).get("suggested_draft"),
        "final": final,
        "tone_notes": tone_notes,
    }

    print("\n--- FINAL OUTPUT KEYS ---")
    print(list(result.keys()))
    print("\n--- FINAL SNIPPET ---")
    print(extract_text_field(result.get("final") or result.get("writer_draft"))[:1600])

    out_path = Path("results/main_run_output.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info("Wrote output to %s", out_path)

    try:
        run_meta = {"run_id": getattr(settings, "RUN_ID", ""), "brief": brief, "timestamp": time.time(), "notes": "auto-run"}
        metrics = evaluate_and_persist((fc_resp or {}).get("fact_check", {}) or {}, run_meta, out_dir="./eval_out")
        log.info("Evaluation metrics: %s", metrics)
    except Exception:
        log.exception("Evaluation failed")

if __name__ == "__main__":
    main()
