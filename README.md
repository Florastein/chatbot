# Personal Assistant

A fully local desktop assistant in Python + Tkinter. A Naive Bayes intent
model recognizes requests, extracts details (app names, filenames, task
descriptions), and calls Python tools to act on your machine. No Ollama,
no network, no third-party Python packages. Works on Windows and Linux.

## Start

Requires Python 3.12+ with Tkinter. No pip dependencies.

```powershell
# Windows
python gui_chatbot.py

# Linux
python3 gui_chatbot.py
```

## Use

Chat naturally. The assistant classifies each message into an intent and
either acts immediately or asks a follow-up for missing details, so
multi-turn requests work:

- "open notepad" / "launch calculator" — launch whitelisted apps
- "find requirements.txt" / "search for my report" — filename search
- "save a note" → "buy milk" — notes
- "add a task" → "finish homework" — to-dos
- "what time is it" — date and time
- "system info" — platform and Python version
- "list directory" / "what's in this folder"
- "check this: <attached code>" — Python syntax + definitions

Slash commands: `/note TEXT`, `/todo TEXT`, `/notes`, `/todos`, `/time`,
`/open notepad`, `/open calculator`, `/help`. Enter sends; Shift+Enter
adds a line.

Other tabs: Notes and To-dos save immediately (select a task to toggle
completion); Files & Apps searches filenames in a chosen folder and opens
whitelisted apps. Attach text to chat pastes a UTF-8 text/source file
under 24 KB into the draft; it is inspected, not executed.

## How it works

- `local_model.py` — multinomial Naive Bayes intent classifier, trained
  on `intents.json`; never touches the network. `data/intent_model.json`
  is regenerated automatically when the intents change.
- `tool_actions.py` — tool registry and cross-platform actions.
- `custom_assistant.py` — intent dispatch, parameter extraction, and
  conversation state that links follow-up messages to a pending action.
- `assistant_core.py` — JSON-free flavored services (Store, helpers).
- `gui_chatbot.py` — Tkinter desktop UI.

## Storage

Data lives in `data/assistant.db`, an unencrypted SQLite database beside
the app. Keep the project folder private and back it up as needed.

The assistant opens only whitelisted apps, searches filenames, reads and
parses attached Python (syntax only), and manages notes/tasks. It does
not execute generated code or run arbitrary shell commands.

## Verify

```powershell
python -m unittest -v
```