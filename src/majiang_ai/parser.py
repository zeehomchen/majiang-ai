from __future__ import annotations

import re
from collections import Counter

from .models import SUITS, Tile


TILE_CODE_RE = re.compile(r"^([1-9])([mpsz])$")


def parse_tile_code(raw: str) -> Tile:
    raw = raw.strip().lower()
    match = TILE_CODE_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"Invalid tile code: {raw}")
    rank = int(match.group(1))
    suit = match.group(2)
    if suit == "z" and rank > 7:
        raise ValueError(f"Honor tile rank out of range: {raw}")
    return Tile(rank=rank, suit=suit)


def parse_compact_tiles(raw: str) -> list[Tile]:
    tiles: list[Tile] = []
    digits: list[str] = []
    for char in raw.strip().lower():
        if char.isdigit():
            digits.append(char)
            continue
        if char.isspace():
            continue
        if char not in SUITS:
            raise ValueError(f"Unexpected character in compact hand: {char}")
        if not digits:
            raise ValueError(f"Missing digits before suit marker {char}")
        for digit in digits:
            rank = int(digit)
            if char == "z" and rank > 7:
                raise ValueError(f"Honor tile rank out of range: {digit}{char}")
            tiles.append(Tile(rank=rank, suit=char))
        digits.clear()
    if digits:
        raise ValueError("Dangling digits without suit marker at end of hand string")
    return tiles


def parse_tile_list(raw: str) -> list[Tile]:
    raw = raw.strip()
    if not raw:
        return []
    if any(separator in raw for separator in [",", " "]):
        tiles: list[Tile] = []
        for chunk in re.split(r"[\s,]+", raw):
            if not chunk:
                continue
            if len(chunk) == 2:
                tiles.append(parse_tile_code(chunk))
            else:
                tiles.extend(parse_compact_tiles(chunk))
        return tiles
    return parse_compact_tiles(raw)


def tiles_to_codes(tiles: list[Tile]) -> list[str]:
    return [tile.code for tile in tiles]


def sort_tiles(tiles: list[Tile]) -> list[Tile]:
    return sorted(tiles, key=lambda tile: ("mpsz".index(tile.suit), tile.rank))


def tiles_to_compact(tiles: list[Tile]) -> str:
    groups: dict[str, list[str]] = {suit: [] for suit in "mpsz"}
    for tile in sort_tiles(tiles):
        groups[tile.suit].append(str(tile.rank))
    result: list[str] = []
    for suit in "mpsz":
        if groups[suit]:
            result.append("".join(groups[suit]) + suit)
    return "".join(result)


def count_codes(tiles: list[Tile]) -> Counter[str]:
    return Counter(tile.code for tile in tiles)
