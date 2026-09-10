# Personal Assistant

**Your machine. Your model. Your everyday assistant.**

Personal Assistant turns familiar requests into useful local actions. Save a thought before you lose it, check your task list, find a document, open a utility, or inspect Python source from a single desktop workspace.

The application runs on a custom-trained intent model built with Python's standard library. Its training examples are yours to inspect, edit, and expand. No pretrained weights, inference servers, API keys, subscriptions, or third-party Python packages are required.

The goal is practical: an assistant whose behavior you can understand, whose data stays on your computer, and whose abilities grow through deliberate training and tool development.

> This is an intent-driven assistant with structured responses and registered tools. It performs supported workflows; it is not a general-purpose conversational or code-generating language model.

## What You Can Do

| Capability | Example request | Result |
| --- | --- | --- |
| Save notes | `save a note` | A follow-up collects the text to store. |
| Create tasks | `add a task` | A follow-up collects the task description. |
| Retrieve notes | `show my notes` | Saved notes appear in the conversation. |
| Retrieve tasks | `show my tasks` | The assistant returns your task list. |
| Open utilities | `open notepad` | A supported app is launched through an allowlist. |
| Search filenames | `find report.txt` | The assistant attempts to extract search details and asks for missing information. |
| Browse directories | `list directory` | Directory entries are listed. |
| Check the clock | `what time is it` | The current local date, time, and timezone are returned. |
| Check the environment | `system info` | The platform, Python version, and machine architecture are shown. |
| Inspect Python | Attach a Python source file | Syntax errors, functions, and classes are reported. |
| Exchange short messages | `good morning bestie` | Recognized conversational intents receive predefined responses. |

Recognition is approximate. Rephrase an unrecognized request, use a slash command where available, or use the dedicated controls.

## Getting Started

### Requirements

- Python **3.12 or newer**.
- Tkinter and an interactive graphical desktop session.
- A writable project location for saved data and the model cache.
- Windows or Linux; desktop integration depends on installed applications and handlers.

There is no `pip install` step or GPU requirement. Training and inference use local resources.

### Windows

Open a terminal in the project directory:

```powershell
python --version
python -m tkinter
python gui_chatbot.py
```

The Tkinter diagnostic opens a small window. Close it before launching the assistant.

The supplied launcher can also be run from the project directory:

```powershell
.\run.bat
```

If Windows recognizes `py` instead of `python`, use `py` for the equivalent commands.

### Linux

From the project directory:

```bash
python3 --version
python3 -m tkinter
python3 gui_chatbot.py
```

Alternatively:

```bash
bash run.sh
```

Some distributions package Tkinter separately. If the diagnostic reports a missing module, install the matching Tkinter package for your Python installation.

File-manager integration uses `xdg-open`. Linux utility launching depends on desktop handlers; the current Notepad and Calculator mappings may not work on every desktop.

### First Launch

The assistant loads a cached intent model or trains one automatically when the cache is missing or the dataset has changed. No model download or remote connection is needed.

Notes, tasks, chat messages, and settings are saved locally during use.

## Your First Workflow

```text
You: save a note
Assistant: What should I save in the note?

You: The demo needs a working file-search example.
Assistant: Note saved.

You: add a task
Assistant: What task should I add?

You: Prepare the demo before Friday.
Assistant: Task added.

You: show my tasks
```

A pending request connects the next message to the missing detail. Say `cancel` to abandon the workflow and start again.

A date inside a task description remains text. It does not schedule an alarm or notification.

## The Desktop Experience

### Chat

Use Chat for requests, follow-up answers, and supported conversational exchanges. Saved messages remain available between launches.

- **Enter** sends a message.
- **Shift+Enter** inserts a new line.
- **Clear chat** deletes saved conversation history after confirmation.
- Messages are limited to 26,000 characters.

Assistant requests run in background threads, with results delivered to the interface through queues.

### Notes and Tasks

Dedicated views provide direct controls for adding and managing saved items. Tasks can be marked complete or reopened. Deleting a selected item requires confirmation.

Use `/notes` or `/todos` to inspect saved content as text in the conversation.

### Files & Apps

Choose a folder and enter all or part of a filename. This view provides explicit control over the search location.

Search is case-insensitive and matches filenames, not document contents. It skips common dependency directories, including `.git`, `node_modules`, `.venv`, and `__pycache__`, as well as directory symlinks and junctions.

To bound the work, search stops at 200 matches or 50,000 examined files. Results can be revealed in the operating system's file manager.

The app controls launch supported utilities. Additional applications require explicit allowlist changes.

### Python Inspection

Attach a UTF-8 Python source file of at most 24,000 bytes, then send the draft for inspection.

The inspector reports syntax errors with line numbers. For successfully parsed source, it lists functions and classes. The attachment is parsed, never executed.

Valid syntax does not establish correct runtime behavior. Dependency problems, logic errors, and security issues require separate testing or review.

## Direct Commands

Slash commands provide a predictable route to common actions.

| Command | Purpose |
| --- | --- |
| `/note TEXT` | Save the supplied text as a note. |
| `/todo TEXT` | Add the supplied text as a task. |
| `/notes` | List saved notes. |
| `/todos` | List saved tasks. |
| `/time` | Show the local date and time. |
| `/open notepad` | Request the supported text-editor launch. |
| `/open calculator` | Request the supported calculator launch. |
| `/help` | Display command help. |

For example:

```text
/note Keep the demo short and show one complete workflow.
/todo Test file search before the presentation.
/notes
```

Use plain `cancel` to leave a natural-language follow-up.

## How the Model Works

The classifier uses **multinomial Naive Bayes**. Training counts words associated with each intent and calculates smoothed word likelihoods. During prediction, those learned weights rank the possible intents for a message.

All training starts from local examples in `intents.json`. No pretrained language-model weights are imported.

Request handling proceeds through a small sequence:

1. Check for attachments, cancellation, and pending conversation state.
2. Tokenize the message and score known intents.
3. Apply explicit phrase-based fallbacks where appropriate.
4. Extract required parameters or ask for missing details.
5. Call a registered tool or return a predefined conversational response.

The classifier requires sufficient known-word coverage, a normalized top score of at least 0.55, and a margin of at least 0.15 over the next candidate. These scores guide routing; they are not guarantees of correctness.

Tokenization currently focuses on English letters, digits, and simple apostrophe forms. Emoji-only messages, unfamiliar spelling, and short replies remain difficult cases.

## Make It Understand Your Phrasing

Edit an intent's `patterns` in `intents.json` to add realistic ways you would ask for that action.

An individual entry follows this format:

```json
{
  "tag": "greeting",
  "patterns": [
    "Hello",
    "Hey there",
    "Good morning"
  ],
  "responses": [
    "Hello! What would you like to get done?"
  ],
  "context": [""]
}
```

This illustrates one entry, not a replacement for the complete dataset.

For useful training data:

- Keep each intent focused on one meaning.
- Add varied examples instead of repeating nearly identical sentences.
- Include abbreviations and casual phrasing you actually use.
- Keep tags unique and provide nonempty training examples.
- Avoid assigning the same tokenized example to different intents.
- Keep action requests distinct from conversational acknowledgments.
- Evaluate unseen requests as well as training examples.

The `context` field is retained in the dataset format. Follow-up behavior lives in conversation-state logic; changing that field alone does not define a workflow.

Restart after editing the dataset. A source fingerprint detects changes and invalidates the model cache.

To train explicitly through the current model API:

```bash
python -c "from local_model import train_and_save; train_and_save(); print('Local model trained.')"
```

Use `python3` where that is your Python command.

### Personal Chat as Inspiration

The dataset includes curated phrasing inspired by a local chat export. This is a deliberate adaptation of examples, not continuous learning from every conversation.

The running assistant does not ingest WhatsApp exports automatically, train itself on saved messages, or reproduce a chat participant's identity.

When adapting personal messages, retain reusable phrasing and remove names, contact details, codes, links, and private disclosures. Keep raw exports out of shared copies of the project.

## Privacy and Data

Classification, training, storage, and Python inspection run locally. Prompts are not sent to an inference service.

Messages, notes, tasks, and settings are stored in `data/assistant.db`, an **unencrypted SQLite database**. Anyone with file access may be able to read it.

Close the application before backing up or restoring the database. Protect backups with the same care as the original. Clearing chat preserves notes and tasks and is not a secure-erasure mechanism.

The model cache, `data/intent_model.json`, can be regenerated from the dataset. It does not contain a backup of your saved notes or tasks.

Actions are limited to implemented tools and allowlisted application launches. The assistant does not execute generated code or arbitrary shell commands. Applications it opens retain their own behavior and network access.

## Current Limitations

- Conversation uses intent matching and predefined responses, not unrestricted text generation.
- Coding support covers Python syntax and definitions, not code generation or comprehensive debugging.
- Tasks are a checklist, not scheduled reminders.
- File search checks names rather than document contents.
- Pending conversation state is not restored after restarting.
- Stored chat history is a transcript, not semantic long-term memory.
- Natural-language parameter extraction and short-message recognition can fail.
- Desktop app launching is platform-dependent and limited to supported utilities.

More training examples can improve recognition of existing capabilities. A new action also needs an implementation and dispatch logic.

## Add a Capability

Extend the assistant by connecting language examples to a working action:

1. Define a distinct intent and representative patterns.
2. Implement and register a tool with clear input parameters.
3. Add parameter extraction and prompts for missing information.
4. Test successful requests, invalid inputs, cancellation, and failures.
5. Retrain and verify the full interaction.

Keep operating-system differences in the desktop-action layer. Worker threads must return results through queues rather than modify Tkinter widgets.

Useful tests check observable outcomes: whether an item was saved correctly, whether the expected tool was called, and whether unsupported input avoided an action.

## Verification

Run the automated suite from the project directory:

```bash
python -m unittest -v
```

Use `python3 -m unittest -v` on Linux where needed.

The suite covers persistence, file-search boundaries, attachments, tool registration, parameter extraction, and assistant workflows. Passing tests do not establish general conversational accuracy or compatibility with every desktop environment.

For a manual check, save and list a note, create a task, search a chosen folder, and attach a small Python file. Restart and confirm that saved information remains.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Python is not found | Verify installation; try `py` on Windows or `python3` on Linux. |
| Tkinter cannot be imported | Install or enable Tkinter for the Python version being used. |
| The GUI cannot open a display | Run from an interactive desktop session. |
| A request is not recognized | Rephrase, use a direct command, or add examples and restart. |
| The assistant expects the wrong detail | Say `cancel`, then begin again. |
| Search returns no matches | Check the folder, filename, exclusions, and search limits. |
| An attachment is rejected | Use UTF-8 text within the size limit and without binary content. |
| A Linux utility does not open | Check `xdg-open` and desktop handlers; mappings may need adaptation. |
| Data cannot be saved | Check write permissions and whether another process has locked the database. |
| Console output fails on a filename | Use a UTF-8 terminal or set `PYTHONIOENCODING=utf-8`. |

## Where It Goes Next

The most useful next steps are stronger short-phrase recognition, better parameter extraction, broader evaluation on unseen requests, and more carefully tested local tools.

The guiding idea stays the same: an assistant you can train, inspect, and shape around the work you actually do.
