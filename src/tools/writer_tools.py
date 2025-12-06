# src/tools/writer_tools.py
from typing import Any, Optional, Dict
import json
import logging
from langchain_core.tools import Tool
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate
from openai import OpenAI
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)
if not logger.handlers:
    setup_logger()


def _safe_loads(s: str):
    try:
        return json.loads(s)
    except:
        return None

def write_draft_tool_factory(llm: Any, name="write_draft") -> Tool:
    prompt = PromptTemplate(
    input_variables=["brief", "outline", "context"],
    template=(
        "You are a Kerala Ayurveda content writer. Use ONLY the BRIEF, OUTLINE and CONTEXT provided below.\n"
        "Do NOT invent facts or use external knowledge.\n\n"
        "CONTEXT: The context will be provided as one or more labeled source blocks like [S1] DOC:docA SECTION:sec1\\n<text>\n\n"
        "BRIEF:\n{brief}\n\nOUTLINE:\n{outline}\n\nCONTEXT:\n{context}\n\n"
        "INSTRUCTIONS (must follow exactly):\n"
        "1) Write the article in Markdown. For ANY factual sentence (history, composition, uses, precautions, product facts), place an inline citation tag such as [S1] immediately after the sentence.\n"
        "2) At the end, return STRICT JSON ONLY with exactly two keys: \"draft\" (the article markdown string) and \"citations\" (an array mapping S-tags to sources).\n"
        "   Example JSON shape (RETURN EXACTLY THIS FORMAT, no extra text):\n"
        "{{\"draft\":\"<article markdown>\", \"citations\":[{{\"s\":\"S1\",\"doc_id\":\"docA\",\"section\":\"sec1\",\"snippet\":\"first 200 chars...\"}}]}}\n"
        "3) The citations array must include one entry per cited tag used in the draft. If you cannot support a factual claim from CONTEXT, append <!-- UNSUPPORTED --> to that sentence and DO NOT invent a citation.\n"
        "4) Produce at least 150 words in the \"draft\".\n"
        "5) Temperature should be 0.0 when calling the LLM (set in chain/llm config).\n"
        )
    )


    chain = LLMChain(llm=llm, prompt=prompt)

    def _tool(inp: str):
        data = _safe_loads(inp) or {}
        out = chain.invoke({
            "brief": data.get("brief", ""),
            "outline": data.get("outline", ""),
            "context": data.get("context", ""),
        })
        parsed = _safe_loads(out) or {"draft": out}
        return parsed


    return Tool(name=name, description="Write draft", func=_tool)