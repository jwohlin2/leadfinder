"""Per-company research budget.

Every company gets a hard ceiling on searches, pages and LLM rounds, plus a
wall-clock cap so one pathological site cannot stall the batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class Budget:
    max_search_queries: int = 20
    max_pages: int = 30
    max_llm_rounds: int = 4
    timeout: float = 420.0
    started: float = field(default_factory=time.monotonic)
    search_results_per_query: int = 12
    queries: list[str] = field(default_factory=list)
    pages_spent: int = 0
    llm_rounds: int = 0
    stop_reason: str = ""

    # -- accounting ------------------------------------------------------
    @property
    def queries_used(self) -> int:
        return len(self.queries)

    @property
    def remaining_pages(self) -> int:
        return max(0, self.max_pages - self.pages_spent)

    @property
    def remaining_queries(self) -> int:
        return max(0, self.max_search_queries - self.queries_used)

    @property
    def remaining_rounds(self) -> int:
        return max(0, self.max_llm_rounds - self.llm_rounds)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    # -- spending --------------------------------------------------------
    def spend_query(self, query: str) -> bool:
        if self.remaining_queries <= 0 or self.timed_out:
            return False
        self.queries.append(query)
        return True

    def spend_pages(self, count: int = 1) -> None:
        self.pages_spent += max(0, count)

    def spend_round(self) -> bool:
        if self.remaining_rounds <= 0 or self.timed_out:
            return False
        self.llm_rounds += 1
        return True

    @property
    def timed_out(self) -> bool:
        return self.elapsed > self.timeout

    def exhausted(self) -> bool:
        return (
            self.remaining_queries <= 0
            or self.remaining_pages <= 0
            or self.remaining_rounds <= 0
            or self.timed_out
        )

    def note_stop(self, reason: str) -> None:
        if not self.stop_reason:
            self.stop_reason = reason
