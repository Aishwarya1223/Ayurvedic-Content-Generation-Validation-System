# src/agents/tone_editor.py
from typing import Any, Dict
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate
from src.utils.logger import setup_logger
from src.utils.config import settings
logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()

TONE_PROMPT_TEMPLATE = (
    "You are a Kerala Ayurveda editor. Use ONLY the DRAFT provided below.\n"
    "Do not add new factual claims. Focus on applying the Kerala Ayurveda brand tone: warm, respectful, and non-prescriptive.\n\n"
    "DRAFT:\n{draft}\n\n"
    "INSTRUCTIONS (must follow exactly):\n"
    "1) Return STRICT JSON and NOTHING ELSE. The JSON object must have two keys:\n"
    '   "edited" -> (string) the edited draft in Markdown.\n'
    '   "notes"  -> (object) a short metadata object, e.g. {{"added_safety_note": true, "edits_summary":"..."}}\n'
    "2) If you cannot produce valid JSON, return EXACTLY this JSON (no extra text):\n"
    '{{"edited":"", "notes":{{"added_safety_note":false,"edits_summary":"invalid_json"}}}}\n'
    "3) Keep the edited draft concise but complete. Do not include any explanation or extra commentary.\n"
)

TONE_EDIT_PROMPT = PromptTemplate(
    input_variables=["draft"],
    template=TONE_PROMPT_TEMPLATE
)
PROMPT_FOR_TONE_EDITOR = (
    "You are a Kerala Ayurveda tone editor. Use ONLY the DRAFT provided below.\n"
    "Do not add new factual claims. Apply the Kerala Ayurveda brand tone: warm, respectful, precise, and non-prescriptive.\n\n"
    "DRAFT TO EDIT:\n{draft}\n\n"
    "TONE REQUIREMENTS (do not deviate):\n"
    "- Warm, reassuring, respectful, and precise.\n"
    "- Non-medical and non-prescriptive (no dosages, no guarantees).\n"
    "- No exaggeration or absolute claims.\n"
    "- Introduce Sanskrit terms gently when used.\n"
    "- Keep paragraphs short; use simple headings when helpful.\n\n"
    "SAFETY RULES:\n"
    "- Preserve factual meaning; do NOT invent claims, benefits, or dosage.\n"
    "- Preserve all citations exactly (e.g., [doc#section]).\n"
    "- If a gentle safety note is missing, add this footer exactly once at the end:\n"
    '  "Individuals with medical conditions, pregnancy, or ongoing medication should consult a qualified healthcare provider before starting any new supplement or therapy."\n\n'
    "INSTRUCTIONS (MUST FOLLOW EXACTLY):\n"
    "1) Return STRICT JSON ONLY and NOTHING ELSE. The JSON object MUST have exactly two keys:\n"
    '   \"edited\" -> (string) the edited draft in Markdown.\n'
    '   \"notes\"  -> (object) metadata, e.g. {{\"added_safety_note\": true/false, \"edits_summary\": \"one-sentence summary\"}}\n'
    "2) If you cannot produce valid JSON, return EXACTLY this JSON (no extra text):\n"
    '{{"edited":"", "notes":{{"added_safety_note":false,"edits_summary":"invalid_json"}}}}\n'
    "3) Keep the edited draft concise but complete. Do not include any explanation or commentary outside the JSON.\n"
    "4) Try to preserve sentence-level citations; do not remove or alter [S#] tags.\n"
    "5) Use temperature 0.0 when calling the LLM for this task.\n"
)

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