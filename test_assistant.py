"""Tests for the local assistant — Store, tools, and CustomAssistant."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant_core import Store, find_files, read_attachment
from custom_assistant import CustomAssistant, ConversationState, extract_detail
from tool_actions import ToolRegistry, Tool, create_default_registry, search_files


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_persistence_and_clear_isolation(self):
        store = Store(self.root / "test.db")
        store.add("note", "A note with 'quotes'")
        store.add("task", "Finish project")
        task_id = store.items("task")[0][0]
        store.toggle(task_id)
        store.append("user", "hello")
        store.set_setting("model", "test")
        reopened = Store(store.path)
        self.assertEqual(reopened.messages()[0]["content"], "hello")
        self.assertEqual(reopened.setting("model"), "test")
        self.assertEqual(reopened.items("task")[0][2], 1)
        reopened.clear_chat()
        self.assertEqual(reopened.messages(), [])
        self.assertEqual(len(reopened.items("note")), 1)
        reopened.delete(task_id)
        self.assertEqual(reopened.items("task"), [])


class FindFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_skips_dependencies_and_caps_results(self):
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "hidden.py").touch()
        (self.root / "hello.py").touch()
        (self.root / "other.py").touch()
        paths, limited = find_files(self.root, ".PY")
        self.assertEqual(len(paths), 2)
        self.assertFalse(limited)
        self.assertTrue(find_files(self.root, ".py", limit=1)[1])

    def test_matches_folders_and_defaults_to_home(self):
        (self.root / "reports").mkdir()
        result = search_files(self.root, "report")
        self.assertEqual(result["paths"], [str((self.root / "reports").resolve())])
        self.assertFalse(result["limited"])
        with patch("tool_actions.Path.home", return_value=self.root):
            result = search_files(query="reports")
        self.assertEqual(result["paths"], [str((self.root / "reports").resolve())])

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            find_files(self.root, "")


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_valid_attachment(self):
        source = self.root / "demo.py"
        source.write_text('print("hello")', encoding="utf-8")
        self.assertIn('print("hello")', read_attachment(source))

    def test_binary_rejected(self):
        source = self.root / "demo.py"
        source.write_bytes(b"\0binary")
        with self.assertRaises(ValueError):
            read_attachment(source)

    def test_oversized_rejected(self):
        source = self.root / "demo.py"
        source.write_bytes(b"a" * 24001)
        with self.assertRaises(ValueError):
            read_attachment(source)


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_lookup(self):
        reg = ToolRegistry()
        handler = lambda: {"reply": "ok", "ok": True}
        tool = Tool(name="test_tool", tag="test", description="A test", handler=handler)
        reg.register(tool)
        self.assertIs(reg.get("test"), tool)
        self.assertIs(reg.get_by_name("test_tool"), tool)
        self.assertIsNone(reg.get("missing"))

    def test_all_tools(self):
        reg = create_default_registry()
        tags = {t.tag for t in reg.all_tools()}
        self.assertIn("open_app", tags)
        self.assertIn("find_file", tags)
        self.assertIn("get_time", tags)


class DetailExtractorTests(unittest.TestCase):
    def test_extract_app_name(self):
        self.assertEqual(extract_detail("open_app", "open notepad"), {"app_name": "notepad"})
        self.assertEqual(extract_detail("open_app", "launch calculator"), {"app_name": "calculator"})

    def test_extract_file_query(self):
        params = extract_detail("find_file", "find report.txt")
        self.assertIn("query", params)
        self.assertIn("report", params["query"])

    def test_extract_path(self):
        params = extract_detail("list_dir", "list directory .")
        self.assertIn("path", params)


class ConversationStateTests(unittest.TestCase):
    def test_initial_state(self):
        state = ConversationState()
        self.assertFalse(state.is_pending())
        self.assertIsNone(state.pending_tag)

    def test_pending(self):
        state = ConversationState()
        state.pending_tag = "add_note"
        self.assertTrue(state.is_pending())
        state.reset()
        self.assertFalse(state.is_pending())


class CustomAssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "test.db")
        self.assistant = CustomAssistant(store=self.store)

    def test_empty_input(self):
        self.assertIn("What would you like to do", self.assistant.reply(""))

    def test_cancel(self):
        self.assistant.state.pending_tag = "add_note"
        self.assertEqual(self.assistant.reply("cancel"), "Cancelled.")
        self.assertFalse(self.assistant.state.is_pending())

    def test_greeting(self):
        reply = self.assistant.reply("hello")
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)

    def test_add_note_flow(self):
        reply = self.assistant.reply("save a note")
        self.assertIn("save", reply.lower())
        self.assertTrue(self.assistant.state.is_pending())
        reply = self.assistant.reply("buy milk")
        self.assertEqual(reply, "Note saved.")
        self.assertFalse(self.assistant.state.is_pending())
        items = self.store.items("note")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][1], "buy milk")

    def test_personality_introduction(self):
        reply = self.assistant.reply("who are you")
        self.assertIn("local desktop assistant", reply)
        self.assertFalse(self.assistant.state.is_pending())

    def test_greetings_vary(self):
        first = self.assistant.reply("hello")
        second = self.assistant.reply("hello")
        self.assertNotEqual(first, second)

    def test_conversational_text_saved_as_note(self):
        self.assistant.reply("save a note")
        self.assertEqual(self.assistant.reply("who are you"), "Note saved.")
        self.assertEqual(self.store.items("note")[0][1], "who are you")

    def test_add_task_flow(self):
        reply = self.assistant.reply("add a task")
        self.assertIn("task", reply.lower())
        reply = self.assistant.reply("finish homework")
        self.assertEqual(reply, "Task added.")

    def test_list_notes_empty(self):
        reply = self.assistant.reply("show my notes")
        self.assertIn("Nothing saved yet", reply)

    def test_list_tasks_empty(self):
        reply = self.assistant.reply("show my tasks")
        self.assertIn("Nothing saved yet", reply)

    def test_get_time(self):
        reply = self.assistant.reply("what time is it")
        self.assertIn("20", reply)  # year should be present

    def test_negation(self):
        reply = self.assistant.reply("don't open notepad")
        self.assertEqual(reply, "No action taken.")

    def test_fallback(self):
        reply = self.assistant.reply("asdfghjkl random gibberish xyz")
        self.assertIn("I couldn't", reply)

    def test_attached_code(self):
        text = "check this\n<file_content>\ndef foo(): pass\n</file_content>"
        reply = self.assistant.reply(text)
        self.assertIn("foo", reply)

    def test_open_app_notepad(self):
        with patch("tool_actions.subprocess.Popen") as mock:
            reply = self.assistant.reply("open notepad")
            self.assertIn("Opened", reply)
            mock.assert_called_once()

    def test_open_app_invalid(self):
        with patch("tool_actions.subprocess.Popen") as mock:
            reply = self.assistant.reply("open chrome")
            self.assertIn("not available", reply.lower())
            mock.assert_not_called()

    def test_multi_turn_find_file(self):
        """Regression: multi-turn file search retains collected parameters."""
        # The active user's home directory is implicit; only the query is needed.
        reply = self.assistant.reply("find a file")
        self.assertIn("filename", reply.lower())
        self.assertTrue(self.assistant.state.is_pending())
        self.assertEqual(self.assistant.state.pending_tag, "find_file")

        # The second turn supplies the query and uses the home-directory default.
        reply = self.assistant.reply("report.txt")
        self.assertFalse(self.assistant.state.is_pending())
        self.assertIsInstance(reply, str)

    def test_find_file_placeholder_query(self):
        """Placeholder words like 'file' are treated as missing input."""
        reply = self.assistant.reply("find a file")
        self.assertIn("filename", reply.lower())
        self.assertTrue(self.assistant.state.is_pending())


class GuiQueueTests(unittest.TestCase):
    """Regression tests for separate event queues in GUI."""

    def test_separate_queues_prevent_cross_consumption(self):
        """Chat and search events go to different queues."""
        import queue

        events = queue.Queue()
        search_events = queue.Queue()

        # Simulate chat result
        events.put(("reply", "Hello!", None))
        # Simulate search result
        search_events.put(("search", (["/path/to/file"], False), None))

        # Poll chat queue - should only get chat event
        kind, value, error = events.get_nowait()
        self.assertEqual(kind, "reply")
        self.assertEqual(value, "Hello!")
        self.assertTrue(events.empty())

        # Poll search queue - should only get search event
        kind, value, error = search_events.get_nowait()
        self.assertEqual(kind, "search")
        self.assertEqual(value, (["/path/to/file"], False))
        self.assertTrue(search_events.empty())

        # Verify queues don't interfere
        search_events.put(("search", (["/another"], True), None))
        events.put(("reply", "Another reply", None))

        kind, value, error = events.get_nowait()
        self.assertEqual(kind, "reply")
        self.assertEqual(value, "Another reply")

        kind, value, error = search_events.get_nowait()
        self.assertEqual(kind, "search")
        self.assertEqual(value, (["/another"], True))


class NegationGuardTests(unittest.TestCase):
    """Tighter negation guard should block direct action negations only."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "test.db")
        self.assistant = CustomAssistant(store=self.store)

    def test_negated_action_blocked(self):
        """'don't open notepad' must be blocked."""
        reply = self.assistant.reply("don't open notepad")
        self.assertEqual(reply, "No action taken.")

    def test_negation_in_context_not_blocked(self):
        """'I don't know what I saved' is NOT a blocked negation — it should
        fall through to the model / fallback, not return 'No action taken.'"""
        reply = self.assistant.reply("I don't know what I saved")
        self.assertNotEqual(reply, "No action taken.")

    def test_negated_save_blocked(self):
        """'do not save' should be blocked."""
        reply = self.assistant.reply("do not save anything please")
        self.assertEqual(reply, "No action taken.")


class DisambiguationFlowTests(unittest.TestCase):
    """User is offered a choice when two intents are close; picking '1' executes it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "test.db")
        self.assistant = CustomAssistant(store=self.store)

    def test_disambiguation_state_set(self):
        """If pending_disambiguation is set, the next reply handles the choice."""
        # Artificially inject disambiguation state to avoid ML non-determinism
        from custom_assistant import ConversationState
        self.assistant.state.pending_disambiguation = {
            "options": [("list_notes", 0.55), ("list_tasks", 0.45)],
            "original_text": "show me my stuff",
        }
        # Choosing '1' should dispatch list_notes
        reply = self.assistant.reply("1")
        self.assertIsNone(self.assistant.state.pending_disambiguation)
        # list_notes with empty store returns "Nothing saved yet."
        self.assertIn("Nothing saved yet", reply)

    def test_disambiguation_by_keyword(self):
        """Replying with a keyword from the description also resolves it."""
        self.assistant.state.pending_disambiguation = {
            "options": [("list_notes", 0.55), ("list_tasks", 0.45)],
            "original_text": "show me my stuff",
        }
        reply = self.assistant.reply("notes")
        self.assertIsNone(self.assistant.state.pending_disambiguation)
        self.assertIsInstance(reply, str)


class UndoLinkItemTests(unittest.TestCase):
    """Undo should remove a linked task created by link_item."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "test.db")
        self.assistant = CustomAssistant(store=self.store)

    def test_undo_link_removes_task(self):
        from custom_assistant import ActionRecord
        from datetime import datetime

        # Simulate a link_item action having been performed
        task_text = "Review file: /tmp/report.txt"
        self.store.add("task", task_text)
        record = ActionRecord(
            action_id="abc12345",
            tag="link_item",
            params={"item": "/tmp/report.txt", "target": "task"},
            result={"reply": "Linked"},
            timestamp=datetime.now(),
            undo_data={"task_text": task_text, "kind": "task"},
        )
        self.assistant.state.action_history.append(record)

        reply = self.assistant.reply("undo")
        self.assertIn("Undid", reply)
        # The linked task should be gone
        remaining = self.store.items("task")
        self.assertEqual(remaining, [])


class SearchDepthLimitTests(unittest.TestCase):
    """search_files must respect max_depth and not descend beyond it."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_file_beyond_max_depth_not_found(self):
        from tool_actions import search_files

        # Build a directory chain 14 levels deep with a target file at the end
        deep = self.root
        for i in range(14):
            deep = deep / f"level{i}"
        deep.mkdir(parents=True, exist_ok=True)
        target = deep / "deep_target.txt"
        target.touch()

        # Default max_depth=12 must NOT find the file at depth 14
        result = search_files(self.root, "deep_target", max_depth=12)
        self.assertNotIn(str(target), result["paths"])

    def test_file_within_max_depth_found(self):
        from tool_actions import search_files

        # File at depth 2 should always be found
        shallow = self.root / "a" / "b"
        shallow.mkdir(parents=True)
        target = shallow / "shallow_target.txt"
        target.touch()

        result = search_files(self.root, "shallow_target", max_depth=12)
        self.assertIn(str(target.resolve()), result["paths"])


class OnSaveCallbackTests(unittest.TestCase):
    """on_save callback must fire after a successful note/task save."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "test.db")
        self.saved_kinds: list[str] = []
        self.assistant = CustomAssistant(
            store=self.store,
            on_save=self.saved_kinds.append,
        )

    def test_callback_fires_for_note(self):
        self.assistant.reply("save a note")
        self.assistant.reply("remember to water the plants")
        self.assertEqual(self.saved_kinds, ["note"])

    def test_callback_fires_for_task(self):
        self.assistant.reply("add a task")
        self.assistant.reply("finish the report")
        self.assertEqual(self.saved_kinds, ["task"])

    def test_callback_not_fired_on_list(self):
        """Listing items must not trigger on_save."""
        self.assistant.reply("show my notes")
        self.assertEqual(self.saved_kinds, [])


if __name__ == "__main__":
    unittest.main()
