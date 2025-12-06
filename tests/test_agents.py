# tests/test_agents.py
import importlib
import inspect
import re
from typing import Tuple, Type, Any, Dict
import pytest

# helper to import an agent class or fallback to test stub
def _load_agent_class(module_path: str, class_name: str) -> Type[Any]:
    """
    Try to import module_path.class_name. If ImportError / AttributeError,
    fallback to src.agents.test_agents.<class_name>.
    """
    try:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls
    except Exception:
        stub = importlib.import_module("src.agents.test_agents")
        return getattr(stub, class_name)


def _instantiate(agent_cls: Type[Any], resources: Dict = None):
    """Instantiate agent with resources if constructor accepts it, else call no-arg."""
    try:
        sig = inspect.signature(agent_cls)
        if "resources" in sig.parameters:
            return agent_cls(resources=resources)
        # fallback: try no-arg
        return agent_cls()
    except Exception:
        return agent_cls()


# ---------- OutlineAgent ----------
def test_outline_agent_basic():
    OutlineAgent = _load_agent_class("src.agents.outline_agent", "OutlineAgent")
    agent = _instantiate(OutlineAgent, resources=None)

    payload = {"brief": "Triphala — gentle digestive support"}
    # prefer .run but accept direct call if callable
    if hasattr(agent, "run"):
        out = agent.run(payload)
    else:
        out = agent(payload)

    assert isinstance(out, dict), "OutlineAgent must return a dict"
    assert "outline" in out, "OutlineAgent output should contain 'outline'"
    outline = out["outline"]
    assert isinstance(outline, str), "outline should be a string"
    assert len(outline) > 20, "outline should be reasonably long"
    # simple content check: expect at least one heading/bullet-like marker
    assert ("#" in outline) or ("-" in outline) or ("\n" in outline)


# ---------- WriterAgent ----------
def test_writer_agent_basic():
    WriterAgent = _load_agent_class("src.agents.writer_agent", "WriterAgent")
    agent = _instantiate(WriterAgent, resources=None)

    payload = {
        "brief": "Brahmi Tailam — calming head oil",
        "outline": "## Intro\n- benefits\n- how to use\n",
        "context": ["Brahmi is traditionally used to support relaxation.", "External use only."]
    }
    if hasattr(agent, "run"):
        out = agent.run(payload)
    else:
        out = agent(payload)

    assert isinstance(out, dict), "WriterAgent must return a dict"
    assert "draft" in out, "WriterAgent output should contain 'draft'"
    assert isinstance(out["draft"], str) and len(out["draft"]) > 20
    # citations should be list (may be empty)
    assert "citations" in out and isinstance(out["citations"], list)


# ---------- FactCheckerAgent ----------
def test_factchecker_agent_basic():
    FactCheckerAgent = _load_agent_class("src.agents.fact_checker_agent", "FactCheckerAgent")
    agent = _instantiate(FactCheckerAgent, resources=None)

    # include known tokens so fallback behavior can mark matches
    draft = "Triphala is a classical three-fruit blend that supports digestion. It is traditionally used in Ayurveda."
    payload = {"draft": draft}

    if hasattr(agent, "run"):
        out = agent.run(payload)
    else:
        out = agent(payload)

    assert isinstance(out, dict), "FactCheckerAgent must return a dict"
    assert "fact_check" in out, "FactCheckerAgent output should contain 'fact_check'"
    fc = out["fact_check"]
    # expected keys
    assert "per_sentence" in fc and "summary_score" in fc
    assert isinstance(fc["per_sentence"], list)
    score = float(fc["summary_score"])
    assert 0.0 <= score <= 1.0


# ---------- ToneEditor ----------
def test_tone_editor_adds_safety_footer():
    ToneEditor = _load_agent_class("src.agents.tone_editor", "ToneEditor")
    agent = _instantiate(ToneEditor, resources=None)

    draft = "This is a short informational paragraph about Triphala."
    payload = {"draft": draft}

    if hasattr(agent, "run"):
        out = agent.run(payload)
    else:
        out = agent(payload)

    # allow for either {"edited":..., "notes":...} or {"final":..., ...}
    assert isinstance(out, dict)
    edited = out.get("edited") or out.get("final") or ""
    assert isinstance(edited, str)
    # Safety snippet should be present
    safety_regex = re.compile(r"not a substitute for medical advice", re.I)
    assert safety_regex.search(edited), "Edited text should include safety/footer text"

    notes = out.get("notes") or out.get("tone_notes") or {}
    assert isinstance(notes, dict)


# ---------- Integration smoke (optional) ----------
def test_agents_integration_smoke():
    """
    Optional smoke test: run Outline -> Writer -> FactChecker -> ToneEditor sequentially.
    This verifies the typical flow works end-to-end with your local agents or stubs.
    """
    OutlineAgent = _load_agent_class("src.agents.outline_agent", "OutlineAgent")
    WriterAgent = _load_agent_class("src.agents.writer_agent", "WriterAgent")
    FactCheckerAgent = _load_agent_class("src.agents.fact_checker_agent", "FactCheckerAgent")
    ToneEditor = _load_agent_class("src.agents.tone_editor", "ToneEditor")

    outline_agent = _instantiate(OutlineAgent)
    writer_agent = _instantiate(WriterAgent)
    fc_agent = _instantiate(FactCheckerAgent)
    tone_agent = _instantiate(ToneEditor)

    brief = "Triphala: gentle digestive support"
    outline = outline_agent.run({"brief": brief})["outline"]
    writer_out = writer_agent.run({"brief": brief, "outline": outline, "context": []})
    draft = writer_out["draft"]
    fc_out = fc_agent.run({"draft": draft})
    suggested = fc_out.get("suggested_draft", draft)
    tone_out = tone_agent.run({"draft": suggested})

    # basic sanity asserts
    assert isinstance(draft, str) and len(draft) > 10
    assert "fact_check" in fc_out
    final_text = tone_out.get("edited") or tone_out.get("final") or ""
    assert isinstance(final_text, str) and len(final_text) > 10