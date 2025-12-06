# src/agents/fact_checker_agent.py
from typing import Any, Dict
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_classic.agents import initialize_agent, AgentType

from src.tools.fact_checker_tools import (
    fact_check_tool_factory,
    format_report_tool_factory,
)
from src.utils.config import settings
logger = logging.getLogger(__name__)

class FactCheckerAgent:

    def __init__(self, llm: Any = None, llm_model_name: str = "gpt-4o", verbose: bool = False):
        if llm is None:
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=llm_model_name,
                temperature=0.0
            )
        else:
            self.llm = llm

        self.verbose = verbose

    def _parse_tool_output(self, out):
        """Normalize tool/agent output into a dict."""
        if isinstance(out, dict):
            return out
        if isinstance(out, str):
            s = out.strip()
            if not s:
                return {}
            try:
                return json.loads(s)
            except Exception:
                if s.startswith('"') and s.endswith('"'):
                    try:
                        return json.loads(s.strip('"'))
                    except Exception:
                        pass
                return {"raw": s}
        try:
            return json.loads(str(out))
        except Exception:
            return {"raw": str(out)}

    def _build_agent(self, vectordb, embed, bm25, tokenized_docs):
        tools = [
            fact_check_tool_factory(vectordb, embed, bm25=bm25, tokenized_docs=tokenized_docs),
            format_report_tool_factory(llm=self.llm),
        ]

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=10,
            handle_parsing_errors=True,
        )
        return agent

    def run(self, draft_text: str, vectordb: Any, embed: Any,
        bm25=None, tokenized_docs=None, top_k: int = 3, threshold: float = 0.25) -> Dict[str, Any]:

        logger.info("FactCheckerAgent.run: starting fact-check")

        if vectordb is None or embed is None:
            logger.error(
                "FactCheckerAgent.run: missing vectordb or embed (vectordb=%s, embed=%s)",
                vectordb, embed
            )
            return {
                "fact_check": {"per_sentence": [], "summary_score": 0.0, "corrected_draft": draft_text},
                "editor_report": {},
                "suggested_draft": draft_text,
            }

        agent = self._build_agent(vectordb, embed, bm25, tokenized_docs)
        fact_check_instr = (
            "CALL_TOOL fact_check\n"
            f"{{\"draft\": {json.dumps(draft_text)} }}\n"
            "END_CALL\n"
            "Return ONLY the JSON from the tool."
        )

        try:
            fc_raw = agent.run(fact_check_instr)
            fc = self._parse_tool_output(fc_raw)
        except Exception as e:
            logger.error("fact_check tool failed: %s", e)
            return {
                "fact_check": {"per_sentence": [], "summary_score": 0.0, "corrected_draft": draft_text},
                "editor_report": {},
                "suggested_draft": draft_text,
            }
        if not isinstance(fc, dict):
            fc = {"raw": str(fc)}
        format_instr = (
            "CALL_TOOL format_report\n"
            f"{{\"draft\": {json.dumps(draft_text)}, \"fact_check\": {json.dumps(fc)} }}\n"
            "END_CALL\n"
            "Return ONLY the JSON from the tool."
        )

        try:
            fr_raw = agent.run(format_instr)
            report = self._parse_tool_output(fr_raw)
        except Exception as e:
            logger.warning("format_report tool failed: %s", e)
            report = {
                "summary": "",
                "grounding_score": fc.get("summary_score", 0.0) if isinstance(fc, dict) else 0.0,
                "unsupported_sentences": [],
                "suggested_draft": fc.get("corrected_draft", draft_text) if isinstance(fc, dict) else draft_text,
            }

        suggested = report.get("suggested_draft") if isinstance(report, dict) else draft_text

        return {
            "fact_check": fc,
            "editor_report": report,
            "suggested_draft": suggested,
        }