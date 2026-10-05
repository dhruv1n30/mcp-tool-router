"""Measure a router on labelled questions: does it keep the right tool, and how
much tool-schema payload does it save?"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp_tool_router.router import ToolRouter, describe_tool


@dataclass(frozen=True)
class Case:
    """One labelled question. Recall needs every expected tool to be selected."""

    question: str
    expected: tuple[str, ...]


@dataclass
class Report:
    cases: int
    hits: int
    total_tools: int
    mean_selected: float
    payload_all: int
    mean_payload_selected: float
    misses: list[tuple[Case, tuple[str, ...]]] = field(default_factory=list)
    # Questions that matched nothing. With on_no_match="all" they still count as
    # hits (the whole catalog was sent), so recall alone hides them.
    no_match: int = 0

    @property
    def recall(self) -> float:
        return self.hits / self.cases if self.cases else 0.0

    @property
    def payload_reduction(self) -> float:
        return 1 - self.mean_payload_selected / self.payload_all if self.payload_all else 0.0

    def format(self, show_misses: int = 10) -> str:
        lines = [
            f"cases            {self.cases}",
            f"recall           {self.recall:.1%}  ({self.hits}/{self.cases} kept every tool)",
            f"tools per query  {self.mean_selected:.1f} of {self.total_tools} on average",
            f"schema payload   {self.payload_reduction:.0%} smaller"
            f"  (~{self.mean_payload_selected / 4:,.0f} vs ~{self.payload_all / 4:,.0f} tokens)",
            f"no match         {self.no_match} questions matched no tool",
        ]
        for case, missing in self.misses[:show_misses]:
            lines.append(f"  miss: {case.question!r} dropped {', '.join(missing)}")
        if len(self.misses) > show_misses:
            lines.append(f"  ... and {len(self.misses) - show_misses} more misses")
        return "\n".join(lines)


def payload_size(tool: Any) -> int:
    """Approximate serialized size of a tool definition, in characters.

    Characters / 4 is the usual rough token estimate; exact counts need the
    target model's tokenizer.
    """
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump(mode="json", exclude_none=True)
    return len(json.dumps(tool, default=str, sort_keys=True))


def load_tools(path: str | Path) -> list[Any]:
    """Load a tool catalog: a JSON list, or a ``tools/list`` result ``{"tools": [...]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tools"), list):
        data = data["tools"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of tools or an object with a 'tools' list")
    return data


def _expected(value: Any, where: str) -> tuple[str, ...]:
    names = value if isinstance(value, list) else str(value or "").split("|")
    names = tuple(str(name).strip() for name in names if str(name).strip())
    if not names:
        raise ValueError(f"{where}: no expected tool given")
    return names


def load_cases(path: str | Path) -> list[Case]:
    """Load labelled questions from CSV or JSONL.

    CSV needs ``question`` and ``expected`` columns; JSONL lines need the same
    keys. ``expected`` is one tool name, several joined by ``|``, or (JSONL) a list.
    """
    path = Path(path)
    cases: list[Case] = []
    if path.suffix.lower() == ".jsonl":
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "question" not in row or "expected" not in row:
                raise ValueError(f"{path}:{number}: need 'question' and 'expected' keys")
            cases.append(Case(str(row["question"]), _expected(row["expected"], f"{path}:{number}")))
    else:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"question", "expected"} <= set(reader.fieldnames):
                raise ValueError(f"{path}: CSV needs 'question' and 'expected' columns")
            for number, row in enumerate(reader, 2):
                cases.append(Case(row["question"], _expected(row["expected"], f"{path}:{number}")))
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def evaluate(router: ToolRouter, cases: Sequence[Case]) -> Report:
    """Run every case through ``router.select`` and summarise recall and payload."""
    names = {describe_tool(tool)[0] for tool in router.tools}
    unknown = sorted({n for case in cases for n in case.expected} - names)
    if unknown:
        raise ValueError(f"Expected tools not in the catalog: {unknown}")

    sizes = {describe_tool(tool)[0]: payload_size(tool) for tool in router.tools}
    hits, no_match, selected_counts, payloads, misses = 0, 0, [], [], []
    for case in cases:
        no_match += not router.rank(case.question)
        selected = {describe_tool(tool)[0] for tool in router.select(case.question)}
        missing = tuple(name for name in case.expected if name not in selected)
        if missing:
            misses.append((case, missing))
        else:
            hits += 1
        selected_counts.append(len(selected))
        payloads.append(sum(sizes[name] for name in selected))

    return Report(
        cases=len(cases),
        hits=hits,
        total_tools=len(names),
        mean_selected=sum(selected_counts) / len(cases) if cases else 0.0,
        payload_all=sum(sizes.values()),
        mean_payload_selected=sum(payloads) / len(cases) if cases else 0.0,
        misses=misses,
        no_match=no_match,
    )
