from typing import Any, Dict
import json
import logging

from openai import OpenAI
from langchain_openai import ChatOpenAI

from src.tools.writer_tools import write_draft_tool_factory, _safe_loads
from src.tools.outline_tools import create_outline_tool_factory
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()


class WriterAgent:
    def __init__(self, llm: Any = None, model_name: str = "gpt-4o", verbose: bool = False):
        if llm is None:
            try:
                llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=model_name, temperature=0.0)
            except Exception:
                llm = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm = llm
        self.verbose = verbose
        self._outline_tool = create_outline_tool_factory(self.llm)
        self._write_tool = write_draft_tool_factory(self.llm)

    def run(self, brief: str, context: str = "") -> Dict[str, Any]:
        logger.info("WriterAgent.run: brief len=%d", len(brief or ""))
        outline_input = {"brief": brief}
        try:
            outline_resp = self._outline_tool.func(json.dumps(outline_input, ensure_ascii=False))
        except Exception as e:
            logger.exception("outline tool failed: %s", e)
            outline_resp = self._outline_tool.func(outline_input)

        outline_json = _safe_loads(outline_resp) or (outline_resp if isinstance(outline_resp, dict) else {})
        outline_text = outline_json.get("outline", "") if isinstance(outline_json, dict) else ""

        write_input = {"brief": brief, "outline": outline_text, "context": context}
        try:
            draft_resp = self._write_tool.func(json.dumps(write_input, ensure_ascii=False))
        except Exception as e:
            logger.exception("write tool failed: %s", e)
            draft_resp = self._write_tool.func(write_input)

        draft_json = _safe_loads(draft_resp) or (draft_resp if isinstance(draft_resp, dict) else {})
        draft_text = draft_json.get("draft", "") if isinstance(draft_json, dict) else ""

        return {"outline": outline_text, "draft": draft_text, "raw_outline_resp": outline_resp, "raw_draft_resp": draft_resp}