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


if __name__ == "__main__":
    unittest.main()
