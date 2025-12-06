from typing import Any, Dict
import json
import logging

from openai import OpenAI
from langchain_openai import ChatOpenAI

from src.tools.outline_tools import create_outline_tool_factory
from src.utils.logger import setup_logger
from src.utils.config import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()


class OutlineAgent:
    """Simple outline runner that calls the outline tool directly (no AgentExecutor)."""

    def __init__(self, llm: Any = None, model_name: str = "gpt-4o", verbose: bool = False):
        if llm is None:
            try:
                llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=model_name, temperature=0.0)
            except Exception:
                llm = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm = llm
        self.tool = create_outline_tool_factory(llm=self.llm)

    def run(self, brief: str) -> Dict[str, Any]:
        logger.info("OutlineAgent.run: brief len=%d", len(brief or ""))
        payload = {"brief": brief}
        try:
            resp = self.tool.func(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.exception("outline tool failed: %s", e)
            resp = self.tool.func(payload)

        # tool is expected to return a dict; if it's a JSON string, try parsing.
        if isinstance(resp, str):
            try:
                parsed = json.loads(resp)
                if isinstance(parsed, dict):
                    resp = parsed
            except Exception:
                pass

        outline = ""
        if isinstance(resp, dict):
            outline = resp.get("outline", "") or ""
        else:
            outline = str(resp)

        return {"outline": outline, "raw": resp}
