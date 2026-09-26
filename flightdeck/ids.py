"""Sortable identifiers.

ULIDs rather than UUIDs because turn ids are read in logs and sorted by time constantly;
lexicographic order matching chronological order is worth the 26 characters.
"""

from __future__ import annotations

import os
import time
from threading import Lock

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = Lock()
_last_ms = 0
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def ulid(now_ms: int | None = None) -> str:
    """Return a 26-char ULID, monotonic within a process even inside the same millisecond."""
    global _last_ms, _last_rand
    ms = now_ms if now_ms is not None else int(time.time() * 1000)
    with _lock:
        if ms == _last_ms:
            _last_rand += 1
            rand = _last_rand
        else:
            rand = int.from_bytes(os.urandom(10), "big")
            _last_ms, _last_rand = ms, rand
    return _encode(ms, 10) + _encode(rand & ((1 << 50) - 1), 10) + _encode(rand >> 50, 6)


def ulid_time_ms(value: str) -> int:
    """Extract the millisecond timestamp encoded in a ULID."""
    ms = 0
    for char in value[:10]:
        ms = (ms << 5) | _CROCKFORD.index(char.upper())
    return ms
