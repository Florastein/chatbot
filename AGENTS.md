# AGENTS.md

## What this is

A fully local desktop assistant (Tkinter GUI). A Naive Bayes intent model
(built with stdlib only) recognizes requests, extracts details (app
names, filenames, task text), and calls Python tools. No Ollama, no
network, no third-party pip packages — do not reintroduce external
dependencies or remote inference.

## Commands

- Run tests: `python -m unittest -v` (there is no lint/format/typecheck config)
- Launch GUI: `python gui_chatbot.py` (also `run.bat` on Windows, `run.sh` on Linux)
- Must run from the repo root: modules are flat top-level files (not a
  package), so imports like `from local_model import ...` only work when
  the root is on `sys.path`. The path contains spaces and parentheses.

## Architecture

- `local_model.py` — multinomial Naive Bayes intent classifier. Trains on
  `intents.json`; caches to `data/intent_model.json`, revalidated via a
  `source_hash` fingerprint. Editing `intents.json` triggers a retrain on
  next load. `data/` is gitignored, so a fresh clone retrains at startup.
- `tool_actions.py` — `Tool` (name/tag/parameters/handler) + `ToolRegistry`
  mapping intent tag → tool, plus cross-platform actions. App launching is
  whitelist-only (notepad, calculator) via `_app_command` (Windows:
  System32 exe; Linux: `xdg-open`).
- `custom_assistant.py` — `CustomAssistant`: classify intent → extract
  parameters → call tool. `ConversationState` links follow-up messages to
  a pending action (multi-turn). NB is weak on short or out-of-vocabulary
  input, so `_guess_intent` provides regex fallbacks for greetings,
  open/find/time/system-info phrasing.
- `assistant_core.py` — legacy-exposed services: `Store` (SQLite:
  `messages`, `items`, `settings` tables), `find_files`,
  `read_attachment`, and re-exports `CustomAssistant`/`open_app`.
- `gui_chatbot.py` — Tkinter UI. Sidebar navigation, canvas-drawn chat
  bubbles (`ChatFeed`), status pill. Uses plain `tk` widgets + the `Palette`
  class for cross-platform colors; ttk/clam is only for `Treeview` and
  `Scrollbar`. Keep this split when restyling.

## Adding a capability

Three places change together: add patterns in `intents.json`, register a
`Tool` in `create_default_registry()` (`tool_actions.py`), and add a
parameter extractor in `extract_detail()` (`custom_assistant.py`).

## Gotchas

- Cross-platform is a hard requirement (Windows + Linux). Don't hardcode
  `explorer.exe`, `SystemRoot`, or Windows paths outside `tool_actions`.
  `gui_chatbot._reveal` branches on `sys.platform`.
- Tk threading: worker threads must not touch Tk widgets. Results go
  through a queue polled by `root.after`.
- Console on Windows is cp1252 by default; printing tree/listing output can
  raise `UnicodeEncodeError` (the repo dir contains a file whose name has
  an emoji). Use UTF-8 output or `PYTHONIOENCODING=utf-8`.
- Legacy artifacts are NOT loaded and shouldn't be wired back in:
  `train_chatbot.py`, `chatbot_model.h5`, `words.pkl`, `classes.pkl`
  (old Keras/NLTK medical chatbot).
- Security invariant: the assistant never executes generated code or
  arbitrary shell commands; only whitelisted actions. Preserve this.
- `data/assistant.db` is unencrypted local storage — keep the folder private.
- Note/test flow: "save a note" → assistant asks "What should I save…" →
  user replies "buy milk" → saved. Test new intents through this
  multi-turn path, not just single messages.