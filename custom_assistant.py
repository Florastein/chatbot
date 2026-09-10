"""Custom local assistant with tool calling and conversation state.

Classifies user intent via the local Naive Bayes model, extracts
parameters from the user message, and dispatches to tools registered
in :mod:`tool_actions`.  Conversation state links follow-up messages
back to the pending action so multi-turn workflows work naturally.
"""
from __future__ import annotations

import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from local_model import load_model, tokens
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


def _extract_index(text: str) -> int | None:
    """Extract a 1-based index from ordinal words or numbers."""
    ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "last": -1,
    }
    lowered = text.casefold()
    for word, idx in ordinals.items():
        if re.search(rf"\b{word}\b", lowered):
            return idx
    # Number
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _extract_new_name(text: str) -> str | None:
    """Extract a new name from 'call it X', 'rename to X', 'name it X'."""
    patterns = [
        r"(?:call|rename|name)\s+(?:it|that|this|the\s+\w+)\s+(?:to\s+)?[\"']([^\"']+)[\"']",
        r"(?:call|rename|name)\s+(?:it|that|this|the\s+\w+)\s+(?:to\s+)?(\S.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_detail(tag: str, text: str) -> dict[str, str]:
    """Return a dict of parameter-name -> value for the given intent tag."""
    extractors: dict[str, dict[str, Any]] = {
        "open_app": {"app_name": _extract_app_name},
        "find_file": {"query": _extract_query, "folder": _extract_folder},
        "list_dir": {"path": _extract_path},
        "add_note": {},
        "add_task": {},
        "select_result": {"index": lambda t: str(_extract_index(t) or "")},
        "link_item": {"index": lambda t: str(_extract_index(t) or ""), "target_kind": _extract_link_target},
        "rename_item": {"index": lambda t: str(_extract_index(t) or ""), "new_name": _extract_new_name},
        "undo_action": {},
    }
    params = extractors.get(tag, {})
    result: dict[str, str] = {}
    for param, fn in params.items():
        val = fn(text)
        if val is not None and val != "":
            result[param] = val
    return result


def _extract_link_target(text: str) -> str | None:
    """Extract what to link to (task, note, project)."""
    lowered = text.casefold()
    if "task" in lowered:
        return "task"
    if "note" in lowered:
        return "note"
    if "project" in lowered:
        return "project"
    return None


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ActionRecord:
    """A single action that can be undone."""
    action_id: str
    tag: str
    params: dict[str, str]
    result: dict[str, Any]
    timestamp: datetime
    undo_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Tracks what the assistant is waiting for and what it has learned."""
    pending_tag: str | None = None
    context: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0
    
    # Search results for reference resolution
    last_search_results: list[str] = field(default_factory=list)
    last_search_tag: str | None = None
    
    # Selected item from search/list
    selected_item_index: int | None = None  # 0-based
    selected_item_kind: str | None = None  # "file", "note", "task", "project"
    selected_item_data: dict[str, Any] = field(default_factory=dict)
    
    # Active project context
    active_project: str | None = None
    active_project_notes: list[int] = field(default_factory=list)  # note IDs
    active_project_tasks: list[int] = field(default_factory=list)  # task IDs
    
    # Action history for undo
    action_history: list[ActionRecord] = field(default_factory=list)
    
    # Disambiguation state
    pending_disambiguation: dict[str, Any] | None = None

    def reset(self) -> None:
        self.pending_tag = None
        self.context.clear()
        self.pending_disambiguation = None

    def is_pending(self) -> bool:
        return self.pending_tag is not None

    def add_action(self, tag: str, params: dict[str, str], result: dict[str, Any], undo_data: dict[str, Any] = None) -> None:
        """Record an action for potential undo."""
        action = ActionRecord(
            action_id=str(uuid.uuid4())[:8],
            tag=tag,
            params=params.copy(),
            result=result.copy(),
            timestamp=datetime.now(),
            undo_data=undo_data or {},
        )
        self.action_history.append(action)
        # Keep last 50 actions
        if len(self.action_history) > 50:
            self.action_history.pop(0)

    def get_last_action(self) -> ActionRecord | None:
        if self.action_history:
            return self.action_history[-1]
        return None

    def pop_last_action(self) -> ActionRecord | None:
        if self.action_history:
            return self.action_history.pop()
        return None

    def set_search_results(self, results: list[str], tag: str) -> None:
        self.last_search_results = results
        self.last_search_tag = tag
        self.selected_item_index = None
        self.selected_item_kind = None
        self.selected_item_data = {}

    def select_result(self, index: int, kind: str = "file", data: dict = None) -> bool:
        """Select a result by 1-based index. Returns True if valid."""
        if 1 <= index <= len(self.last_search_results):
            self.selected_item_index = index - 1
            self.selected_item_kind = kind
            self.selected_item_data = data or {}
            return True
        return False

    def get_selected_item(self) -> dict[str, Any] | None:
        if self.selected_item_index is not None and self.selected_item_index < len(self.last_search_results):
            return {
                "index": self.selected_item_index,
                "path": self.last_search_results[self.selected_item_index],
                "kind": self.selected_item_kind,
                **self.selected_item_data,
            }
        return None

    def clear_selection(self) -> None:
        self.selected_item_index = None
        self.selected_item_kind = None
        self.selected_item_data = {}


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

        # Explicit command handling first (slash commands, explicit patterns)
        explicit = self._try_explicit_command(text)
        if explicit is not None:
            return explicit

        self.state.turn_count += 1

        # If there is a pending action, feed the follow-up text into it
        if self.state.is_pending():
            return self._handle_followup(text)

        # Focused extraction rules for reference resolution, linking, etc.
        focused = self._try_focused_extraction(text)
        if focused is not None:
            return focused

        # Naive Bayes for broader routing
        tag, confidence = self.model.predict(text)
        if tag is None:
            # Verb-based heuristic fallback complements the ML model for
            # out-of-vocabulary app/file names ("open chrome").
            tag = self._guess_intent(text)
            if tag is None:
                return self._fallback_reply()

        # Check for competing interpretations
        competing = self._check_competing_intents(text, tag, confidence)
        if competing:
            return competing

        return self._dispatch(tag, text, confidence)

    def _try_explicit_command(self, text: str) -> str | None:
        """Handle explicit slash commands and very clear patterns first."""
        # Slash commands
        if text.startswith("/"):
            command, _, argument = text.partition(" ")
            if command in ("/note", "/todo"):
                kind = "note" if command == "/note" else "task"
                if not argument.strip():
                    return f"Usage: {command} <text>"
                if self.store is None:
                    return f"(No store) Would save: {argument}"
                self.store.add(kind, argument.strip())
                return "Saved."
            if command in ("/notes", "/todos"):
                kind = "note" if command == "/notes" else "task"
                rows = self.store.items(kind) if self.store else []
                return "\n".join(f"{i}. {'[Done] ' if done else ''}{c}" for i, c, done in rows) or "Nothing saved yet."
            if command == "/time":
                from datetime import datetime
                return datetime.now().astimezone().strftime("%A, %d %B %Y, %H:%M %Z")
            if command == "/open":
                if not argument:
                    return "Usage: /open <app_name>"
                tool = self.registry.get("open_app")
                if tool:
                    return tool.handler(app_name=argument).get("reply", f"Opened {argument}.")
            if command == "/help":
                return (
                    "Commands: /note TEXT, /todo TEXT, /notes, /todos, /time, "
                    "/open notepad, /open calculator, /help, /undo.\n"
                    "Or just chat naturally — I can search files, open apps, "
                    "manage notes/tasks, link items, rename, and undo."
                )
            if command == "/undo":
                return self._handle_undo()
            return "Unknown command. Try /help."

        # Explicit "undo" without slash
        if text.casefold() in {"undo", "undo that", "reverse that", "take it back"}:
            return self._handle_undo()

        return None

    def _try_focused_extraction(self, text: str) -> str | None:
        """Handle reference resolution, linking, renaming via focused rules."""
        lowered = text.casefold()

        # Reference resolution: "the second one", "that file", "number 3"
        if self.state.last_search_results:
            idx = _extract_index(text)
            if idx is not None:
                # Handle negative index (last)
                if idx < 0:
                    idx = len(self.state.last_search_results) + idx + 1
                if self.state.select_result(idx, kind=self.state.last_search_tag or "file"):
                    item = self.state.get_selected_item()
                    if item:
                        return f"Selected: {item['path']}. What would you like to do with it? (open, link to task, attach to chat, etc.)"
                return f"No result at position {idx}. There are {len(self.state.last_search_results)} results."

        # Linking: "link it to my task", "attach to project"
        if self.state.selected_item_index is not None:
            if re.search(r"\blink\b.*\b(task|note|project)\b", lowered) or \
               re.search(r"\b(attach|add)\b.*\bto\b.*\b(task|note|project)\b", lowered):
                return self._dispatch("link_item", text, 0.9)

            # Setting active project: "this is my project", "set as project"
            if re.search(r"\b(this is|set as|make)\b.*\bmy\s+project\b", lowered) or \
               re.search(r"\bproject\b", lowered) and len(lowered.split()) <= 4:
                return self._dispatch("link_item", text, 0.8)

            # Renaming: "call it X", "rename to X"
            if re.search(r"\b(call|rename|name)\b.*\b(it|that|this)\b", lowered):
                return self._dispatch("rename_item", text, 0.9)

        # Renaming via action history: "call it X" after creating a task/note
        if not self.state.selected_item_index:
            if re.search(r"\b(call|rename|name)\b.*\b(it|that|this)\b", lowered):
                # Check if there's a recent add_task, add_note, or link_item action
                for action in reversed(self.state.action_history):
                    if action.tag in ("add_task", "add_note", "link_item"):
                        return self._dispatch("rename_item", text, 0.8)

        # "Open it" / "open that" referring to selected file
        if self.state.selected_item_index is not None and re.search(r"\b(open|launch|run)\b.*\b(it|that|this)\b", lowered):
            item = self.state.get_selected_item()
            if item and item["kind"] in ("file", "find_file"):
                import subprocess, sys
                path = item["path"]
                try:
                    if sys.platform == "win32":
                        os.startfile(path)
                    else:
                        subprocess.Popen(["xdg-open", path])
                    return f"Opened {path}."
                except Exception as e:
                    return f"Could not open: {e}"

        return None

    def _check_competing_intents(self, text: str, primary_tag: str, primary_conf: float) -> str | None:
        """If multiple intents are close in confidence, offer a specific choice."""
        # Get top predictions
        words = tokens(text)
        known = [w for w in words if w in self.model.vocabulary]
        if not known:
            return None
        scores = {tag: sum(self.model.data['weights'][tag].get(w, 0) for w in known)
                  for tag in self.model.data['weights']}
        maximum = max(scores.values())
        probabilities = {tag: math.exp(score - maximum) for tag, score in scores.items()}
        total = sum(probabilities.values())
        ranked = sorted(((p / total, tag) for tag, p in probabilities.items()), reverse=True)
        
        # If top two are close, offer disambiguation
        if len(ranked) >= 2:
            conf1, tag1 = ranked[0]
            conf2, tag2 = ranked[1]
            if conf1 - conf2 < 0.2 and conf2 > 0.3:
                # Store for disambiguation
                self.state.pending_disambiguation = {
                    "options": [(tag1, conf1), (tag2, conf2)],
                    "original_text": text,
                }
                tag1_desc = self._intent_description(tag1)
                tag2_desc = self._intent_description(tag2)
                return f"I'm not sure — did you mean **{tag1_desc}** or **{tag2_desc}**? (Reply '1' or '2', or clarify)"

        return None

    def _intent_description(self, tag: str) -> str:
        tool = self.registry.get(tag)
        if tool:
            return tool.description
        descriptions = {
            "add_note": "save a note",
            "add_task": "add a task",
            "list_notes": "list notes",
            "list_tasks": "list tasks",
            "greeting": "say hello",
            "goodbye": "say goodbye",
            "thanks": "say thanks",
            "options": "show capabilities",
            "personality": "describe myself",
            "check_in": "ask how I am",
            "feeling_ok": "say you're fine",
            "acknowledgment": "acknowledge",
            "apology": "apologize",
            "share_something": "share something",
            "wait": "wait a moment",
            "clarification": "ask for clarification",
            "laughter": "laugh",
            "food_check_in": "ask about food",
        }
        return descriptions.get(tag, tag)

    @staticmethod
    def _guess_intent(text: str) -> str | None:
        """Best-effort intent guess from verbs/phrases when the model is unsure."""
        lowered = text.casefold()
        if re.search(r"\b(?:open|launch|start|run)\b", lowered):
            return "open_app"
        if re.search(r"\b(?:find|search|locate|where is|where did)", lowered):
            return "find_file"
        if re.search(r"\b(?:what time|the time|today's date|what's the date)\b", lowered):
            return "get_time"
        if re.search(r"\b(?:list|what's in|show me what's in|show me|show)\b.*\b(?:dir|directory|folder|files|here)\b", lowered):
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
        # Disambiguation response handling
        if self.state.pending_disambiguation:
            return self._handle_disambiguation(text)

        # New conversation-flow intents
        if tag == "select_result":
            return self._handle_select_result(text)
        if tag == "link_item":
            return self._handle_link_item(text)
        if tag == "rename_item":
            return self._handle_rename_item(text)
        if tag == "undo_action":
            return self._handle_undo()

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

    def _handle_disambiguation(self, text: str) -> str:
        """Handle user's choice from a disambiguation prompt."""
        options = self.state.pending_disambiguation["options"]
        original = self.state.pending_disambiguation["original_text"]
        
        # Try to parse as number
        idx = _extract_index(text)
        if idx is not None and 1 <= idx <= len(options):
            chosen_tag, _ = options[idx - 1]
            self.state.pending_disambiguation = None
            return self._dispatch(chosen_tag, original, 0.7)
        
        # Try to match by keyword
        lowered = text.casefold()
        for i, (tag, _) in enumerate(options):
            desc = self._intent_description(tag).casefold()
            if any(word in lowered for word in desc.split()):
                self.state.pending_disambiguation = None
                return self._dispatch(tag, original, 0.7)
        
        return "Please reply with the number (1, 2, ...) or a keyword from one of the options."

    def _handle_select_result(self, text: str) -> str:
        """Handle explicit selection from search results."""
        idx = _extract_index(text)
        if idx is None:
            return "Which result would you like to select? (e.g., 'the second one', 'number 3')"
        if idx < 0:
            idx = len(self.state.last_search_results) + idx + 1
        if self.state.select_result(idx, kind=self.state.last_search_tag or "file"):
            item = self.state.get_selected_item()
            if item:
                return f"Selected: {item['path']}. What would you like to do with it?"
        return f"No result at position {idx}. There are {len(self.state.last_search_results)} results."

    def _handle_link_item(self, text: str) -> str:
        """Link selected item to a task, note, or project."""
        if self.state.selected_item_index is None:
            return "No item selected. Search for something first, then select it."
        
        item = self.state.get_selected_item()
        if not item:
            return "Selection lost. Please search again."
        
        target_kind = _extract_link_target(text) or "task"
        
        if target_kind == "task":
            # Create or link to a task
            if self.store is None:
                return "(No store) Would link file to task."
            # For now, create a task referencing the file
            task_text = f"Review file: {item['path']}"
            self.store.add("task", task_text)
            self.state.add_action("link_item", {"item": item['path'], "target": "task"}, {"reply": "Linked"}, {"task_text": task_text})
            return f"Created task: \"{task_text}\" linked to {item['path']}."
        
        elif target_kind == "note":
            if self.store is None:
                return "(No store) Would link file to note."
            note_text = f"File reference: {item['path']}"
            self.store.add("note", note_text)
            self.state.add_action("link_item", {"item": item['path'], "target": "note"}, {"reply": "Linked"}, {"note_text": note_text})
            return f"Created note: \"{note_text}\" linked to {item['path']}."
        
        elif target_kind == "project":
            self.state.active_project = item['path']
            self.state.active_project_notes = []
            self.state.active_project_tasks = []
            return f"Set active project to: {item['path']}. Add notes/tasks to build it up."
        
        return f"Don't know how to link to '{target_kind}'. Try task, note, or project."

    def _handle_rename_item(self, text: str) -> str:
        """Rename a task or note."""
        new_name = _extract_new_name(text)
        if not new_name:
            return "What would you like to call it? (e.g., 'call it Prepare project demo')"
        
        # If we have a selected item from search, try to rename the corresponding task/note
        if self.state.selected_item_index is not None:
            item = self.state.get_selected_item()
            if item and self.store:
                # Try to find matching task/note by content
                kind = item.get("kind", "task")
                rows = self.store.items(kind)
                for item_id, content, done in rows:
                    if item['path'] in content or content in item['path']:
                        self.store.update(item_id, new_name)
                        self.state.add_action("rename_item", {"old": content, "new": new_name}, {"reply": "Renamed"}, {"old_id": item_id, "old_content": content, "kind": kind})
                        return f"Renamed {kind} to: {new_name}"
        
        # Fallback: rename last created task/note from action history
        last_action = self.state.get_last_action()
        if last_action and last_action.tag in ("add_note", "add_task", "link_item"):
            # Determine kind from the action
            if last_action.tag == "link_item":
                kind = last_action.undo_data.get("target", "task")
                old_content = last_action.undo_data.get("task_text") or last_action.undo_data.get("note_text", "")
                # Find the item_id for this content
                rows = self.store.items(kind)
                for item_id, content, done in rows:
                    if content == old_content:
                        self.store.update(item_id, new_name)
                        self.state.add_action("rename_item", {"old": old_content, "new": new_name}, {"reply": "Renamed"}, {"old_id": item_id, "old_content": old_content, "kind": kind})
                        return f"Renamed {kind} to: {new_name}"
            else:
                kind = "note" if last_action.tag == "add_note" else "task"
                old_content = last_action.params.get("content", "")
                # Find the item_id for this content
                rows = self.store.items(kind)
                for item_id, content, done in rows:
                    if content == old_content:
                        self.store.update(item_id, new_name)
                        self.state.add_action("rename_item", {"old": old_content, "new": new_name}, {"reply": "Renamed"}, {"old_id": item_id, "old_content": old_content, "kind": kind})
                        return f"Renamed {kind} to: {new_name}"
        
        return "Nothing to rename. Create a task/note first, or select one from search results."

    def _handle_undo(self) -> str:
        """Undo the last action."""
        action = self.state.pop_last_action()
        if not action:
            return "Nothing to undo."
        
        if self.store is None:
            return f"(No store) Would undo: {action.tag} - {action.params}"
        
        try:
            if action.tag in ("add_note", "add_task"):
                kind = "note" if action.tag == "add_note" else "task"
                # Find and delete the most recently added item matching the content
                rows = self.store.items(kind)
                for item_id, content, done in rows:
                    if content == action.params.get("content", ""):
                        self.store.delete(item_id)
                        return f"Undid: deleted {kind} \"{content[:50]}...\""
                return f"Could not find {kind} to undo."
            
            elif action.tag == "link_item":
                # Undo linking by deleting the created task/note
                kind = action.undo_data.get("kind", "task")
                content = action.undo_data.get("task_text") or action.undo_data.get("note_text")
                if content:
                    rows = self.store.items(kind)
                    for item_id, c, done in rows:
                        if c == content:
                            self.store.delete(item_id)
                            return f"Undid: deleted linked {kind} \"{content[:50]}...\""
                return "Could not undo link."
            
            elif action.tag == "rename_item":
                kind = action.undo_data.get("kind", "task")
                old_content = action.undo_data.get("old_content", "")
                new_content = action.params.get("new_name", "")
                item_id = action.undo_data.get("old_id")
                # Update back to old content
                if item_id is not None:
                    self.store.update(item_id, old_content)
                    return f"Undid rename: restored \"{old_content[:50]}...\""
                # Fallback: find by new_content and update
                rows = self.store.items(kind)
                for item_id, content, done in rows:
                    if content == new_content:
                        self.store.update(item_id, old_content)
                        return f"Undid rename: restored \"{old_content[:50]}...\""
                return "Could not undo rename."
            
            return f"Undid {action.tag} (partial support)."
        except Exception as e:
            return f"Undo failed: {e}"

    def _handle_tool_intent(self, tool: Tool, text: str) -> str:
        """Extract parameters and call a tool; ask for missing ones if needed."""
        params = self._resolve_defaults(tool.tag, extract_detail(tool.tag, text))

        # If all required parameters are present, execute
        missing = [p for p in tool.parameters if p not in params]
        if not missing:
            result = self._execute_tool(tool, params)
            # Record action for undo if it modifies state
            if tool.tag in ("add_note", "add_task", "open_app"):
                self.state.add_action(tool.tag, params, {"reply": result})
            # Store search results for reference resolution
            if tool.tag == "find_file" and isinstance(result, dict) and "paths" in result:
                self.state.set_search_results(result["paths"], tool.tag)
            return result

        # Ask for the first missing parameter
        self.state.pending_tag = tool.tag
        self.state.context.update(params)
        return self._ask_missing(tool, missing[0])

    def _resolve_defaults(
        self, tag: str, params: dict[str, str]
    ) -> dict[str, str]:
        """Fill in sensible defaults, including the active user's home folder."""
        if tag == "find_file" and "folder" not in params:
            folder = self.store.setting("folder") if self.store is not None else ""
            params["folder"] = folder if folder and Path(folder).is_dir() else str(Path.cwd())
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

        # Handle disambiguation follow-up
        if self.state.pending_disambiguation:
            return self._handle_disambiguation(text)

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
        # Don't reset state for tools that need follow-up context
        try:
            result = tool.handler(**params)
            # Record action for undo if it modifies state
            if tool.tag in ("add_note", "add_task"):
                self.state.add_action(tool.tag, params, {"reply": result})
            # Store search results for reference resolution
            if tool.tag == "find_file" and isinstance(result, dict) and "paths" in result:
                self.state.set_search_results(result["paths"], tool.tag)
            # Clear pending tag but keep conversation context
            if tool.tag not in ("add_note", "add_task"):
                self.state.pending_tag = None
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
            "index": "Which one? (e.g., 'the second one', 'number 3')",
            "target_kind": "Link to what? (task, note, or project)",
            "new_name": "What would you like to call it?",
        }
        return prompts.get(param, f"Please provide: {param}")

    def _save_item(self, tag: str, content: str) -> str:
        """Save a note or task to the store."""
        self.state.pending_tag = None
        if self.store is None:
            return f"(No store) Would save: {content}"
        kind = "note" if tag == "add_note" else "task"
        try:
            self.store.add(kind, content)
            result = "Note saved." if kind == "note" else "Task added."
            self.state.add_action(tag, {"content": content}, {"reply": result})
            # Add to active project if set
            if self.state.active_project:
                if kind == "note":
                    self.state.active_project_notes.append(1)  # placeholder ID
                else:
                    self.state.active_project_tasks.append(1)
            return result
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
        # Store as search results for reference
        self.state.set_search_results([f"{item_id}. {content}" for item_id, content, _ in rows], f"list_{kind}")
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
        capabilities.extend(["save notes", "manage tasks", "link items", "rename items", "undo"])
        return (
            "I couldn't quite work out what you need. Try a short request, like 'save a note'. "
            f"I can: {', '.join(capabilities)}."
        )
