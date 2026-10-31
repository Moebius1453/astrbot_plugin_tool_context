import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Awaitable, Callable


SummaryFunction = Callable[[dict], Awaitable[str]]


def _encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_encode(value).encode("utf-8")).hexdigest()


class ToolLog:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = asyncio.Lock()

    def _path(self, scope: str) -> Path:
        if not isinstance(scope, str) or not scope:
            raise ValueError("scope_required")
        return self.directory / (_digest(scope) + ".jsonl")

    async def _append(self, scope: str, record: dict) -> None:
        line = _encode(record) + "\n"
        path = self._path(scope)
        async with self._lock:
            await asyncio.to_thread(self._append_sync, path, line)

    def _append_sync(self, path: Path, line: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    async def archive(self, scope: str, tool_call: dict, result: dict) -> str:
        payload = {"tool_call": copy.deepcopy(tool_call), "result": copy.deepcopy(result)}
        reference = _digest({"scope": scope, "payload": payload})
        await self._append(scope, {"kind": "tool", "reference": reference, "payload": payload})
        return reference

    async def save_summary(self, scope: str, reference: str, summary: str) -> None:
        await self._append(scope, {"kind": "summary", "reference": reference, "text": summary})

    async def records(self, scope: str) -> list[dict]:
        path = self._path(scope)
        async with self._lock:
            return await asyncio.to_thread(self._records_sync, path)

    @staticmethod
    def _records_sync(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("invalid_log_record")
        return records

    async def read(self, scope: str, reference: str) -> dict:
        for record in await self.records(scope):
            if record.get("kind") == "tool" and record.get("reference") == reference:
                return copy.deepcopy(record["payload"])
        raise KeyError("unknown_record")


def _complete_groups(messages: list[dict]) -> list[tuple[int, list[dict], list[int]]]:
    groups = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list) or not calls:
            continue
        if any(not isinstance(call, dict) for call in calls):
            continue
        ids = [call.get("id") for call in calls]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            continue
        positions = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            positions.append(cursor)
            cursor += 1
        results = [messages[position].get("tool_call_id") for position in positions]
        if all(isinstance(value, str) for value in results) and sorted(results) == sorted(ids):
            groups.append((index, calls, positions))
    return groups


class ToolContext:
    def __init__(
        self,
        log: ToolLog,
        summarize: SummaryFunction,
        *,
        tools: frozenset[str],
        min_chars: int = 2000,
        max_summary_chars: int = 600,
        timeout: float = 30,
    ):
        if min_chars <= max_summary_chars or max_summary_chars < 1 or timeout <= 0:
            raise ValueError("invalid_limits")
        self.log = log
        self.summarize = summarize
        self.tools = frozenset(tools)
        self.min_chars = min_chars
        self.max_summary_chars = max_summary_chars
        self.timeout = timeout
        self._lock = asyncio.Lock()

    async def project(self, scope: str, messages: list[dict]) -> list[dict]:
        # Work on a frozen copy so new events cannot be covered by an older summary.
        source = copy.deepcopy(messages)
        view = copy.deepcopy(source)
        async with self._lock:
            try:
                records = await self.log.records(scope)
                archived = {record["reference"] for record in records if record.get("kind") == "tool"}
                summaries = {
                    record["reference"]: record["text"]
                    for record in records if record.get("kind") == "summary"
                }
            except (OSError, ValueError, KeyError, TypeError):
                return view
            for index, calls, positions in _complete_groups(source):
                content = source[index].get("content")
                if isinstance(content, list) and any(
                    isinstance(part, dict) and part.get("type") in ("think", "thinking", "redacted_thinking")
                    for part in content
                ):
                    continue
                by_id = {call["id"]: call for call in calls}
                # A later assistant message proves a previous request could see this result.
                eligible = any(message.get("role") == "assistant" for message in source[positions[-1] + 1:])
                for position in positions:
                    call = by_id[source[position]["tool_call_id"]]
                    function = call.get("function")
                    if not isinstance(function, dict) or function.get("name") not in self.tools:
                        continue
                    result = source[position]
                    if not isinstance(result.get("content"), str):
                        continue
                    payload = {"tool_call": call, "result": result}
                    reference = _digest({"scope": scope, "payload": payload})
                    try:
                        if reference not in archived:
                            await self.log.archive(scope, call, result)
                            archived.add(reference)
                        if not eligible or len(result["content"]) < self.min_chars:
                            continue
                        text = summaries.get(reference)
                        if text is None:
                            text = await asyncio.wait_for(self.summarize(copy.deepcopy(payload)), self.timeout)
                            if not self._valid_summary(text):
                                continue
                            text = text.strip()
                            await self.log.save_summary(scope, reference, text)
                            summaries[reference] = text
                        if not self._valid_summary(text):
                            continue
                        replacement = f"{text.strip()} [tool-log:{reference}]"
                        if len(replacement) < len(result["content"]):
                            view[position]["content"] = replacement
                    except (OSError, ValueError, TypeError, KeyError, TimeoutError):
                        continue
                    except Exception:
                        # Provider failures must never discard the original tool result.
                        continue
        return view

    def _valid_summary(self, text: object) -> bool:
        return (
            isinstance(text, str)
            and bool(text.strip())
            and len(text.strip()) <= self.max_summary_chars
            and text.strip().isascii()
            and "\n" not in text.strip()
            and "\r" not in text.strip()
        )
