"""FSVLM — Fine-tune Gemma 4 for visual defect detection. Locally."""

__version__ = "0.1.0"


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports — keep `import fsvlm` fast (<1s, no torch)."""
    if name == "types":
        from fsvlm import types as _mod

        return _mod
    if name == "config":
        from fsvlm import config as _mod

        return _mod
    if name == "exceptions":
        from fsvlm import exceptions as _mod

        return _mod
    if name == "Inspector":
        from fsvlm.agents.inspector_agent import InspectorAgent

        return InspectorAgent
    if name == "InspectorSession":
        from fsvlm.agents.inspector_agent import InspectorSession

        return InspectorSession
    if name == "EventBus":
        from fsvlm.events import EventBus

        return EventBus
    raise AttributeError(f"module 'fsvlm' has no attribute {name!r}")
