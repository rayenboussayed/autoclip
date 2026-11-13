"""Short-lived WebSocket tickets: single-use, 60-second grants scoped to one job."""

from __future__ import annotations

import secrets
import time

_issued: dict[str, tuple[str, float]] = {}  # (job_id, (ticket, expiry))
TTL_SECONDS = 60.0
ALLOWED_ACTIVE = 8


def issue(job_id: str) -> tuple[str, int]:
    """Mint a ticket for ``job_id``; returns ``(ticket, ttl_seconds)``."""
    now = time.time()
    stale = [key for key, (_, expiry) in ((jid, pair) for jid, pair in _issued.items()) if expiry <= now]
    for key in stale:
        del _issued[key]
    # bound the table so a leaked ticket cannot be flooded into memory forever
    if len(_issued) >= ALLOWED_ACTIVE:
        oldest = min(_issued, key=lambda k: _issued[k][1])
        del _issued[oldest]
    ticket = secrets.token_urlsafe(32)
    _issued[job_id] = (ticket, now + TTL_SECONDS)
    return ticket, int(TTL_SECONDS)


def redeem(job_id: str, ticket: str) -> bool:
    """True when ``ticket`` is the unexpired ticket issued for ``job_id`` (single use)."""
    entry = _issued.get(job_id)
    if entry is None:
        return False
    issued_ticket, expiry = entry
    if time.time() > expiry:
        del _issued[job_id]
        return False
    if secrets.compare_digest(ticket, issued_ticket):
        del _issued[job_id]  # single use
        return True
    return False


def reset() -> None:
    """Clear all tickets (test isolation guard)."""
    _issued.clear()
