# src/agents/outline_tools.py
from typing import Any, Optional, Dict
import json
import logging

from langchain_core.tools import Tool
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()


def _safe_loads(s: str) -> Optional[Dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _dict_to_md(d: Dict) -> str:
    parts = []
    title = d.get("title") or d.get("heading")
    if isinstance(title, str):
        parts.append(f"## {title}\n")

    intro = d.get("introduction") or d.get("intro")
    if isinstance(intro, str):
        parts.append(intro + "\n")

    for k, v in d.items():
        if k in ("title", "introduction", "intro"):
            continue
        if isinstance(v, dict):
            h = v.get("heading") or v.get("title") or k
            parts.append(f"### {h}\n")
            if isinstance(v.get("content"), str):
                parts.append(v["content"] + "\n")
        elif isinstance(v, str):
            parts.append(f"### {k}\n{v}\n")

    return "\n".join(parts).strip() or json.dumps(d, ensure_ascii=False)


def create_outline_tool_factory(llm: Any = None, name: str = "create_outline") -> Tool:
    if llm is None:
        llm = OpenAI(temperature=0.2)

    prompt = PromptTemplate(
        input_variables=["brief", "sources"],
        template=(
            "SYSTEM: You are a Kerala Ayurveda content writer. ONLY use the text in the SOURCES below.\n"
            "Do NOT use your pretraining or any outside knowledge. If the required content is NOT present\n"
            "in the SOURCES, reply exactly with the single token: INSUFFICIENT_CONTEXT\n\n"
            "SOURCES (numbered):\n"
            "{sources}\n\n"
            "BRIEF:\n{brief}\n\n"
            "Task: Produce a concise hierarchical outline in Markdown (H2/H3). Each heading that uses a fact\n"
            "must include a source tag like [S1], [S2]. Return STRICT JSON only with this shape:\n"
            '{{\\\"outline\\\":\\\"<markdown outline>\\\"}}'
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    def _tool(input_str: str) -> Dict:
        parsed_in = _safe_loads(input_str) or {}
        brief = parsed_in.get("brief") or parsed_in.get("input") or (input_str if isinstance(input_str, str) else "")
        sources = parsed_in.get("sources") or ""

        try:
            out = chain.invoke({"brief": brief, "sources": sources})
            out_text = out if isinstance(out, str) else getattr(out, "text", None) or str(out)

            parsed_out = _safe_loads(out_text)

            if isinstance(parsed_out, dict) and "outline" in parsed_out:
                val = parsed_out["outline"]
                if isinstance(val, dict):
                    return {"outline": _dict_to_md(val).strip()}
                return {"outline": str(val).strip()}
            if isinstance(out, dict) and "outline" in out:
                val = out["outline"]
                if isinstance(val, dict):
                    return {"outline": _dict_to_md(val).strip()}
                return {"outline": str(val)}

            return {"outline": out_text.strip()}
        except Exception as e:
            logger.exception("create_outline_tool failed: %s", e)
            return {"outline": "", "error": str(e)}

    return Tool(name=name, description="Create an article outline from a brief.", func=_tool)