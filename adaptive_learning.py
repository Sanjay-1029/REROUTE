"""Persistent, verified adaptive outcome memory for REROUTE (M11).

Only terminal, verified outcomes are stored. The JSON store is local to this
machine, versioned, bounded, and written atomically. Set REROUTE_MEMORY_PATH
to choose a different location; set REROUTE_DISABLE_PERSISTENCE=1 for
ephemeral test/demo sessions.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from outcome_verifier import VerificationState

_SCHEMA_VERSION = 1
_MAX_ATTEMPTS = 5000


def _state_values(record) -> set[str]:
    return {getattr(state, "value", str(state)) for state in getattr(record, "verification_states", ())}


class AdaptiveOutcomeMemory:
    """Accumulate verified outcomes and optionally retain them across restarts."""

    def __init__(self, storage_path: str | os.PathLike | None = None, persistent: bool = False) -> None:
        if os.environ.get("REROUTE_DISABLE_PERSISTENCE") == "1":
            persistent = False
        if storage_path is None and persistent:
            storage_path = os.environ.get(
                "REROUTE_MEMORY_PATH",
                str(Path.home() / ".reroute" / "adaptive_memory.json"),
            )
        self.storage_path = Path(storage_path).expanduser() if storage_path else None
        self._counts = defaultdict(lambda: {"success": 0, "failure": 0})
        self._seen_attempts: set[str] = set()
        self._total_ingested = 0
        if self.storage_path:
            self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != _SCHEMA_VERSION:
                return
            counts = payload.get("counts", {})
            if not isinstance(counts, dict):
                return
            for cause, actions in counts.items():
                if not isinstance(actions, dict):
                    continue
                for action, values in actions.items():
                    if not isinstance(values, dict):
                        continue
                    success, failure = values.get("success", 0), values.get("failure", 0)
                    if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (success, failure)):
                        self._counts[(str(cause), str(action))] = {"success": success, "failure": failure}
            attempts = payload.get("seen_attempts", [])
            if isinstance(attempts, list):
                self._seen_attempts = {str(x) for x in attempts[-_MAX_ATTEMPTS:] if isinstance(x, str)}
            total = payload.get("total_verified_outcomes", 0)
            if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                self._total_ingested = total
        except (OSError, ValueError, TypeError, AttributeError):
            # Corrupt/unavailable history must never prevent safe operation.
            return

    def _save(self) -> None:
        if not self.storage_path:
            return
        grouped = {}
        for (cause, action), values in self._counts.items():
            grouped.setdefault(cause, {})[action] = dict(values)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "total_verified_outcomes": self._total_ingested,
            "seen_attempts": sorted(self._seen_attempts)[-_MAX_ATTEMPTS:],
            "counts": grouped,
        }
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.storage_path.parent, delete=False) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, self.storage_path)
        except OSError:
            # Persistence is best-effort; the in-memory controller remains usable.
            try:
                temporary.unlink(missing_ok=True)
            except (UnboundLocalError, OSError):
                pass

    def ingest(self, records: Iterable[object]) -> int:
        added = 0
        for record in records:
            attempt_id = getattr(record, "attempt_id", "")
            if not attempt_id or attempt_id in self._seen_attempts:
                continue
            action = getattr(getattr(record, "decision", None), "action", None)
            action = getattr(action, "value", str(action))
            if action == "REQUEST_HUMAN_HELP":
                continue
            states = _state_values(record)
            cause = getattr(getattr(record, "diagnosis", None), "cause", "UNKNOWN")
            cause = getattr(cause, "value", str(cause))
            outcome = None
            if VerificationState.RECOVERY_FAILED.value in states:
                outcome = "failure"
            elif action == "GATHER_INFORMATION" and VerificationState.OBSERVATION_REFRESHED.value in states:
                outcome = "success"
            elif action == "WAIT" and VerificationState.ENVIRONMENT_CHANGED.value in states:
                outcome = "success"
            elif VerificationState.PROGRESS_RESTORED.value in states or VerificationState.TASK_COMPLETED.value in states:
                outcome = "success"
            elif getattr(record, "succeeded", None) is False:
                outcome = "failure"
            if outcome is None:
                continue
            self._counts[("ALL", action)][outcome] += 1
            self._counts[(cause, action)][outcome] += 1
            self._seen_attempts.add(str(attempt_id))
            self._total_ingested += 1
            added += 1
        if added:
            if len(self._seen_attempts) > _MAX_ATTEMPTS:
                self._seen_attempts = set(sorted(self._seen_attempts)[-_MAX_ATTEMPTS:])
            self._save()
        return added

    def snapshot(self, diagnosis: str | None = None) -> dict:
        key_prefix = diagnosis if diagnosis else "ALL"
        counts = {action: dict(values) for (cause, action), values in self._counts.items() if cause == key_prefix}
        return {
            "source": "PERSISTENT_VERIFIED_RUNTIME_OUTCOMES" if self.storage_path else "VERIFIED_RUNTIME_OUTCOMES",
            "diagnosis": diagnosis or "ALL",
            "action_statistics": counts,
            "observations": sum(v["success"] + v["failure"] for v in counts.values()),
            "total_verified_outcomes": self._total_ingested,
            "persistence_enabled": self.storage_path is not None,
        }

    @property
    def total_verified_outcomes(self) -> int:
        return self._total_ingested
