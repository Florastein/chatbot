"""Custom local assistant with tool calling and conversation state.

Classifies user intent via the local Naive Bayes model, extracts
parameters from the user message, and dispatches to tools registered
in :mod:`tool_actions`.  Conversation state links follow-up messages
back to the pending action so multi-turn workflows work naturally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_model import load_model
from tool_actions import Tool, ToolRegistry, create_default_registry

BASE = Path(__file__).resolve().parent

# Tokens that indicate the user wants to cancel a pending action
_CANCEL_WORDS = frozenset({"cancel", "never mind", "nevermind", "no", "stop"})


# ---------------------------------------------------------------------------
# Parameter extractors
# ---------------------------------------------------------------------------

def _extract_app_name(text: str) -> str | None:
    """Pull an app name from the user message."""
    text_lower = text.casefold()
    for name in ("notepad", "calculator", "nano", "gedit", "code", "vim", "emacs"):
        if name in text_lower:
            return name
    # Fallback: last bare word after open/launch/start/run
    match = re.search(r"\b(?:open|launch|start|run)\s+(\w+)", text_lower)
    if match:
        return match.group(1)
    return None


def _extract_query(text: str) -> str | None:
    """Extract a file search query from the user message."""
    # Strip common prefixes like "find", "search for", "look for"
    cleaned = re.sub(
        r"^(?:find|search(?:\s+for)?|look(?:\s+for)?|where(?:\s+is)?|locate)\s+",
        "", text, flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:the\s+|a\s+|my\s+)", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip("?")
    if not cleaned:
        return None
    # Treat placeholder words as missing input
    if cleaned.casefold() in {"file", "files", "document", "documents", "folder", "folders"}:
        return None
    return cleaned


def _extract_folder(text: str) -> str | None:
    """Extract a folder path from the user message, if present."""
    # Look for quoted paths or known folder names
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    for q in quoted:
        p = Path(q)
        if p.is_dir():
            return str(p)
    # Look for common folder references
    import os
    home = Path.home()
    for name in ("desktop", "documents", "downloads", "home", "tmp", "temp"):
        if name in text.casefold():
            candidate = home / name.capitalize() if name != "home" else home
            if name in ("tmp", "temp"):
                candidate = Path(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")))
            if candidate.is_dir():
                return str(candidate)
    return None


def _extract_path(text: str) -> str | None:
    """Extract or resolve a directory path from the user message."""
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    for q in quoted:
        p = Path(q)
        if p.is_dir():
            return str(p)
    # Bare path segments
    for token in text.split():
        p = Path(token)
        if p.is_dir():
            return str(p)
    return str(Path("."))


def extract_detail(tag: str, text: str) -> dict[str, str]:
    """Return a dict of parameter-name -> value for the given intent tag."""
    extractors: dict[str, dict[str, Any]] = {
        "open_app": {"app_name": _extract_app_name},
        "find_file": {"query": _extract_query, "folder": _extract_folder},
        "list_dir": {"path": _extract_path},
        "add_note": {},   # note content is the entire message
        "add_task": {},   # task content is the entire message
    }
    params = extractors.get(tag, {})
    result: dict[str, str] = {}
    for param, fn in params.items():
        val = fn(text)
        if val is not None:
            result[param] = val
    return result


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Tracks what the assistant is waiting for and what it has learned."""
    pending_tag: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0

    def reset(self) -> None:
        self.pending_tag = None
        self.context.clear()

    def is_pending(self) -> bool:
        return self.pending_tag is not None


# ---------------------------------------------------------------------------
# Custom assistant
# ---------------------------------------------------------------------------

class CustomAssistant:
    """Classify intent, extract details, and call tools."""

    def __init__(self, store: Any = None, registry: ToolRegistry | None = None) -> None:
        self.store = store
        self.model = load_model()
        self.registry = registry or create_default_registry()
        self.state = ConversationState()
        # Load canned responses for non-tool intents (greeting, goodbye, etc.)
        import json
        self.responses: dict[str, list[str]] = {}
        self._response_counts: dict[str, int] = {}
        try:
            intents_path = BASE / "intents.json"
            data = json.loads(intents_path.read_text(encoding="utf-8"))
            for intent in data.get("intents", []):
                self.responses[intent["tag"]] = intent.get("responses", [])
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.state.reset()

    def reply(self, text: str) -> str:
        """Process one user message and return the assistant's reply."""
        text = text.strip()
        if not text:
            return "What would you like to do?"

        # Handle attached file content for code inspection
        if "<file_content>" in text:
            return self._handle_attached_code(text)

        # Handle cancel
        if text.casefold() in _CANCEL_WORDS or text == "/cancel":
            self.state.reset()
            return "Cancelled."

        # Handle negation patterns
        if re.search(r"\b(don't|do not|never)\b", text, re.IGNORECASE):
            return "No action taken."

        self.state.turn_count += 1

        # If there is a pending action, feed the follow-up text into it
        if self.state.is_pending():
            return self._handle_followup(text)

        # Otherwise, classify the intent
        tag, confidence = self.model.predict(text)
        if tag is None:
            # Verb-based heuristic fallback complements the ML model for
            # out-of-vocabulary app/file names ("open chrome").
            tag = self._guess_intent(text)
            if tag is None:
                return self._fallback_reply()

        return self._dispatch(tag, text, confidence)

    @staticmethod
    def _guess_intent(text: str) -> str | None:
        """Best-effort intent guess from verbs/phrases when the model is unsure.

        Naive Bayes is weak on short inputs (single words, rare words), so
        this complementary rule set keeps the common conversational cases
        reliable.
        """
        lowered = text.casefold()
        if re.search(r"\b(?:open|launch|start|run)\b", lowered):
            return "open_app"
        if re.search(r"\b(?:find|search|locate|where is|where did)", lowered):
            return "find_file"
        if re.search(r"\b(?:what time|the time|today's date|what's the date)\b", lowered):
            return "get_time"
        if re.search(r"\b(?:list|what's in|show me what's in)\b.*\b(?:dir|directory|folder|files|here)\b", lowered):
            return "list_dir"
        if re.search(r"\b(?:system info|system information|what os|what platform|system details)\b", lowered):
            return "system_info"
        if re.fullmatch(r"(?:who are you|what is your name|what's your name|describe yourself|what is your personality|what's your personality)[?!.]*", lowered):
            return "personality"
        if re.search(r"\b(?:hi|hello|hey|howdy|good morning|good afternoon|good evening|how are you|what's up)\b", lowered):
            return "greeting"
        if re.search(r"\b(?:thanks|thank you|thx|awesome|great|perfect|sweet)\b", lowered):
            return "thanks"
        if re.search(r"\b(?:what can you do|how can you help|capabilities|features|commands)\b", lowered):
            return "options"
        if re.search(r"\b(?:bye|goodbye|see you|see ya|quit|exit)\b", lowered):
            return "goodbye"
        return None

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, tag: str, text: str, confidence: float) -> str:
        """Route a classified intent to the right handler."""
        # Tool-based intents
        tool = self.registry.get(tag)
        if tool is not None:
            return self._handle_tool_intent(tool, text)

        # Note / task intents — need a follow-up turn for content
        if tag in ("add_note", "add_task"):
            self.state.pending_tag = tag
            prompt = "What should I save in the note?" if tag == "add_note" else "What task should I add?"
            return prompt

        # List intents — read from store directly
        if tag in ("list_notes", "list_tasks"):
            return self._list_items("note" if tag == "list_notes" else "task")

        # Time
        if tag == "get_time":
            return self.registry.get("get_time").handler()["reply"]

        # Canned responses for greeting / goodbye / thanks / etc.
        if tag in self.responses and self.responses[tag]:
            choices = self.responses[tag]
            index = self._response_counts.get(tag, 0)
            self._response_counts[tag] = index + 1
            return choices[index % len(choices)]

        return self._fallback_reply()

    def _handle_tool_intent(self, tool: Tool, text: str) -> str:
        """Extract parameters and call a tool; ask for missing ones if needed."""
        params = self._resolve_defaults(tool.tag, extract_detail(tool.tag, text))

        # If all required parameters are present, execute
        missing = [p for p in tool.parameters if p not in params]
        if not missing:
            return self._execute_tool(tool, params)

        # Ask for the first missing parameter
        self.state.pending_tag = tool.tag
        self.state.context.update(params)
        return self._ask_missing(tool, missing[0])

    def _resolve_defaults(
        self, tag: str, params: dict[str, str]
    ) -> dict[str, str]:
        """Fill in sensible defaults (e.g. the stored search folder)."""
        if tag == "find_file" and "folder" not in params and self.store is not None:
            folder = self.store.setting("folder")
            if folder and Path(folder).is_dir():
                params["folder"] = folder
        return params

    def _handle_followup(self, text: str) -> str:
        """Process a follow-up message for a pending action."""
        tag = self.state.pending_tag
        if tag is None:
            return "Nothing pending."

        # Note / task: the follow-up IS the content
        if tag in ("add_note", "add_task"):
            return self._save_item(tag, text)

        # If this follow-up is actually a *different* confident intent,
        # abandon the pending action rather than hijacking it.
        new_tag, confidence = self.model.predict(text)
        if new_tag and new_tag != tag and confidence >= 0.6:
            self.state.reset()
            return self._dispatch(new_tag, text, confidence)
        guess = self._guess_intent(text)
        if guess and guess != tag:
            self.state.reset()
            return self._dispatch(guess, text, 0.6)

        # Tool follow-up: extract more details and try again
        tool = self.registry.get(tag)
        if tool is None:
            self.state.reset()
            return "I lost track. What did you want to do?"

        params = self._resolve_defaults(tag, extract_detail(tag, text))
        self.state.context.update(params)

        # For notes and tasks, the full text is the content
        if tag == "add_note":
            self.state.context["content"] = text
        elif tag == "add_task":
            self.state.context["content"] = text

        missing = [p for p in tool.parameters if p not in self.state.context]
        if not missing:
            ctx = dict(self.state.context)
            self.state.reset()
            return self._execute_tool(tool, ctx)

        return self._ask_missing(tool, missing[0])

    def _execute_tool(self, tool: Tool, params: dict[str, str]) -> str:
        """Call the tool and return its reply, resetting state."""
        self.state.reset()
        try:
            result = tool.handler(**params)
            return result.get("reply", "Done.")
        except Exception as exc:
            return f"Error: {exc}"

    def _ask_missing(self, tool: Tool, param: str) -> str:
        """Return a prompt asking for the missing parameter."""
        prompts = {
            "app_name": "Which app would you like to open?",
            "query": "What filename or part of a filename should I search for?",
            "folder": "Which folder should I search in?",
            "path": "Which directory should I list?",
            "source": "Paste or attach the Python source to inspect.",
        }
        return prompts.get(param, f"Please provide: {param}")

    def _save_item(self, tag: str, content: str) -> str:
        """Save a note or task to the store."""
        self.state.reset()
        if self.store is None:
            return f"(No store) Would save: {content}"
        kind = "note" if tag == "add_note" else "task"
        try:
            self.store.add(kind, content)
            return "Note saved." if kind == "note" else "Task added."
        except ValueError as exc:
            return str(exc)

    def _list_items(self, kind: str) -> str:
        """List saved notes or tasks."""
        if self.store is None:
            return f"(No store) No {kind}s saved."
        rows = self.store.items(kind)
        if not rows:
            return "Nothing saved yet."
        lines = []
        for item_id, content, done in rows:
            prefix = "[Done] " if kind == "task" and done else ""
            lines.append(f"{item_id}. {prefix}{content}")
        return "\n".join(lines)

    def _handle_attached_code(self, text: str) -> str:
        """Extract attached source and inspect it."""
        start = text.find("<file_content>") + len("<file_content>")
        end = text.rfind("</file_content>")
        if end < start:
            return "The attachment is incomplete."
        source = text[start:end].strip("\n")
        tool = self.registry.get("inspect_code")
        if tool is None:
            return "Code inspection is not available."
        return tool.handler(source=source).get("reply", "Inspection complete.")

    def _fallback_reply(self) -> str:
        """Generic reply when the model is not confident."""
        capabilities = [t.description for t in self.registry.all_tools()]
        capabilities.extend(["save notes", "manage tasks"])
        return (
            "I couldn't quite work out what you need. Try a short request, like 'save a note'. "
            f"I can: {', '.join(capabilities)}."
        )
