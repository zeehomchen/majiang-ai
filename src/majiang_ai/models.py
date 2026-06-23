from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


SUITS = ("m", "p", "s", "z")
NUMBER_SUITS = ("m", "p", "s")
DRAGON_CODES = ("5z", "6z", "7z")
WIND_CODES = ("1z", "2z", "3z", "4z")
ORPHAN_CODES = (
    "1m",
    "9m",
    "1p",
    "9p",
    "1s",
    "9s",
    "1z",
    "2z",
    "3z",
    "4z",
    "5z",
    "6z",
    "7z",
)


@dataclass(frozen=True, slots=True)
class Tile:
    rank: int
    suit: str

    @property
    def code(self) -> str:
        return f"{self.rank}{self.suit}"

    @property
    def is_honor(self) -> bool:
        return self.suit == "z"

    @property
    def is_terminal(self) -> bool:
        return self.suit != "z" and self.rank in (1, 9)

    @property
    def is_terminal_or_honor(self) -> bool:
        return self.is_honor or self.is_terminal


@dataclass(slots=True)
class TableContext:
    visible_counts: Counter[str] = field(default_factory=Counter)
    exposed_triplets: set[str] = field(default_factory=set)
    round_phase: str = "early"
    closed_hand: bool = True


@dataclass(slots=True)
class DiscardOption:
    discard: str
    total_score: float
    base_score: float
    effective_draws: int
    improving_tiles: list[str]
    route_scores: dict[str, float]
    summary: list[str]


@dataclass(slots=True)
class AnalysisResult:
    hand: str
    options: list[DiscardOption]
    visible_counts: Counter[str]
    exposed_triplets: set[str]
