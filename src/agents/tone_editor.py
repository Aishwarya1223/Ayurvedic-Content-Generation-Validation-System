# src/agents/tone_editor.py
from typing import Any, Dict
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate
from src.utils.logger import setup_logger
from src.utils.prompts import PROMPT_FOR_TONE_EDITOR
from src.utils.config import settings
logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()

TONE_EDIT_PROMPT = PromptTemplate(input_variables=["draft"], template=PROMPT_FOR_TONE_EDITOR)

class ToneEditor:

    def __init__(self, llm: Any = None, model_name: str | None = None, temperature: float = 0.0):
        model_name = model_name or settings.CHAT_MODEL_NAME
        if llm is None:
            try:
                self.llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=model_name, temperature=temperature)
            except Exception as e:
                logger.error("Failed to initialize ChatOpenAI: %s. Falling back to placeholder llm.", e)
                self.llm = None
        else:
            self.llm = llm

        self.chain = LLMChain(llm=self.llm, prompt=TONE_EDIT_PROMPT)

    def run(self, draft_text: str):
        logger.info("ToneEditor: running tone edit (draft len=%d)", len(draft_text or ""))

        if isinstance(draft_text, dict):
            for k in ("edited", "suggested_draft", "draft", "text"):
                if k in draft_text and isinstance(draft_text[k], str) and draft_text[k].strip():
                    draft_text = draft_text[k]
                    break
            else:
                vals = [v for v in draft_text.values() if isinstance(v, str) and v.strip()]
                draft_text = vals[0] if vals else json.dumps(draft_text, ensure_ascii=False)

        try:
            try:
                raw_out = self.chain.invoke({"draft": draft_text})
            except Exception:
                raw_out = self.chain.run({"draft": draft_text})
        except Exception as e:
            logger.exception("ToneEditor: LLMChain.run/invoke failed: %s", e)
            return {
                "edited": draft_text,
                "notes": {"added_safety_note": False, "edits_summary": f"error: {e}"},
                "raw_llm_output": "",
            }

        raw_text = raw_out if isinstance(raw_out, str) else getattr(raw_out, "text", None) or str(raw_out)
        raw_text = (raw_text or "").strip()
        logger.error("TONE RAW OUTPUT: %r", raw_text)

        parsed = None
        if raw_text.startswith("{"):
            try:
                parsed = json.loads(raw_text)
            except Exception:
                parsed = None

        if parsed is None:
            import re
            m = re.search(r"\{[\s\S]*\}", raw_text)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            edited = parsed.get("edited", None)
            notes = parsed.get("notes", {})
            if isinstance(edited, str):
                logger.info("ToneEditor: LLM returned valid JSON.")
                return {
                    "edited": edited,
                    "notes": notes if isinstance(notes, dict) else {"notes_raw": notes},
                    "raw_llm_output": raw_text,
                }
        if raw_text and ("\n" in raw_text or "#" in raw_text or "-" in raw_text or len(raw_text) > 30):
            edited_guess = raw_text
            fallback_notes = {
                "added_safety_note": False,
                "edits_summary": "Fallback applied: LLM produced non-JSON output; returned best-effort edited text."
            }
            return {"edited": edited_guess, "notes": fallback_notes, "raw_llm_output": raw_text}

        logger.warning("ToneEditor: LLM returned invalid_json sentinel; returning draft unchanged.")
        return {
            "edited": draft_text,
            "notes": {"added_safety_note": False, "edits_summary": "LLM failed to return JSON; returned original draft."},
            "raw_llm_output": raw_text or '{"error":"invalid_json"}',
        }