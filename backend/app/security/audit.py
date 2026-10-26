"""A real in-memory audit log.

Before this module the security page showed seeded sample rows. These are actual
recorded events: authentication attempts and their outcome, authorization denials,
training lifecycle changes, and security configuration changes. An audit trail that
is not derived from what happened is worse than none, because it looks like
evidence.

Every message is passed through `redaction.redact_text` before storage, so a secret
cannot reach the log even if a caller passes one by mistake. There is a test that
asserts a key placed into an event does not survive.

Bounded by design: a ring buffer of ``maxlen`` entries, so a long-running process
cannot exhaust memory. Events are process-local and are lost on restart, which is a
recorded limitation — a real deployment needs durable, append-only storage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from .redaction import redact_text

#: Cap on retained events. Oldest are discarded first.
DEFAULT_MAXLEN = 500


@dataclass(frozen=True)
class AuditRecord:
    """One recorded event."""

    id: str
    timestamp: datetime
    severity: str
    category: str
    message: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "source": self.source,
        }


class AuditLog:
    """Bounded, ordered, process-local event log."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self._events: deque[AuditRecord] = deque(maxlen=maxlen)
        self._counter = 0

    def record(
        self,
        category: str,
        message: str,
        severity: str = "info",
        source: str = "api",
    ) -> AuditRecord:
        """Append an event. The message is redacted before it is stored."""
        self._counter += 1
        event = AuditRecord(
            id=f"evt-{self._counter}",
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            message=redact_text(message),
            source=redact_text(source),
        )
        self._events.append(event)
        return event

    def record_auth(
        self, outcome: str, label: str | None, path: str, reason: str = ""
    ) -> AuditRecord:
        """Authentication outcome, with the caller identified by label only."""
        who = label or "anonymous"
        severity = "info" if outcome == "granted" else "warning"
        detail = f" ({reason})" if reason else ""
        return self.record(
            category="Access",
            message=f"Authorization {outcome} for {who} on {path}{detail}",
            severity=severity,
            source=who,
        )

    def events(
        self,
        limit: int | None = None,
        severity: str | None = None,
    ) -> list[AuditRecord]:
        """Newest first, optionally filtered."""
        ordered = sorted(self._events, key=lambda event: event.timestamp, reverse=True)
        if severity is not None:
            ordered = [event for event in ordered if event.severity == severity]
        if limit is not None:
            ordered = ordered[:limit]
        return ordered

    def counts(self) -> dict[str, int]:
        """Totals by severity, so the page can summarise without a second pass."""
        totals = {"info": 0, "warning": 0, "critical": 0}
        for event in self._events:
            totals[event.severity] = totals.get(event.severity, 0) + 1
        return totals

    def clear(self) -> None:
        """Drop all events. Used by tests and by an explicit operator action."""
        self._events.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._events)


#: Process-wide log. FastAPI reads it through `app.state`, so a test app can be
#: given its own instance.
audit_log = AuditLog()
