import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("tool_context_core", Path(__file__).with_name("core.py"))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
spec = importlib.util.spec_from_file_location("tool_context_summarizer", Path(__file__).with_name("summarizer.py"))
summarizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarizer)


def messages():
    return [
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "shell", "arguments": '{"command":"run"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 4000},
        {"role": "assistant", "content": "Result received."},
    ]


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = core.ToolLog(Path(self.temp.name))
        self.calls = []

        async def summarize(payload):
            self.calls.append(payload)
            return "Ran the command; its external outcome remains unknown."

        self.context = core.ToolContext(self.log, summarize, tools=frozenset({"shell"}))

    async def test_projection_preserves_source_and_retrieves_original(self):
        source = messages()
        before = copy.deepcopy(source)
        view = await self.context.project("session-a", source)
        self.assertEqual(source, before)
        self.assertEqual(view[0], source[0])
        self.assertEqual(view[1]["tool_call_id"], "c1")
        self.assertLess(len(view[1]["content"]), 1000)
        records = await self.log.records("session-a")
        reference = records[0]["reference"]
        self.assertEqual((await self.log.read("session-a", reference))["result"], source[1])
        with self.assertRaises(KeyError):
            await self.log.read("session-b", reference)

    async def test_recent_result_archived_but_not_replaced(self):
        source = messages()[:2]
        self.assertEqual(await self.context.project("a", source), source)
        self.assertFalse(self.calls)
        self.assertEqual(len(await self.log.records("a")), 1)

    async def test_repeated_requests_reuse_persistent_summary(self):
        first = await self.context.project("a", messages())
        second = await self.context.project("a", messages())
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(await self.log.records("a")), 2)

    async def test_summary_failure_preserves_original(self):
        async def fail(payload):
            raise RuntimeError("provider unavailable")
        self.context.summarize = fail
        source = messages()
        self.assertEqual(await self.context.project("a", source), source)
        self.assertEqual(len(await self.log.records("a")), 1)

    async def test_archive_failure_never_calls_summary(self):
        async def fail(*args):
            raise OSError("disk full")
        self.log.archive = fail
        source = messages()
        self.assertEqual(await self.context.project("a", source), source)
        self.assertFalse(self.calls)

    async def test_incomplete_or_unlisted_tools_untouched(self):
        source = messages()
        source[1]["tool_call_id"] = "other"
        self.assertEqual(await self.context.project("a", source), source)
        source = messages()
        source[0]["tool_calls"][0]["function"]["name"] = "operation_context_run"
        self.assertEqual(await self.context.project("a", source), source)
        self.assertFalse(self.calls)

    async def test_nontext_and_signed_groups_untouched(self):
        source = messages()
        source[1]["content"] = [{"type": "image_url", "image_url": {"url": "image"}}]
        self.assertEqual(await self.context.project("a", source), source)
        source = messages()
        source[0]["content"] = [{"type": "thinking", "signature": "signed"}]
        self.assertEqual(await self.context.project("a", source), source)
        self.assertFalse(self.calls)

    async def test_bad_summary_not_adopted(self):
        for text in ("", "x" * 601, "line one\nline two", None):
            async def bad(payload, value=text):
                return value
            self.context.summarize = bad
            source = messages()
            self.assertEqual(await self.context.project(str(text), source), source)

    async def test_provider_adapter_no_tools_or_chat_history(self):
        class Provider:
            async def text_chat(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"completion_text": "Read a file."})()
        provider = Provider()
        adapter = summarizer.ProviderSummarizer(provider)
        self.assertEqual(await adapter({"result": "sample"}), "Read a file.")
        self.assertEqual(provider.kwargs["contexts"], [])
        self.assertNotIn("func_tool", provider.kwargs)
        self.assertEqual(json.loads(provider.kwargs["prompt"]), {"result": "sample"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
