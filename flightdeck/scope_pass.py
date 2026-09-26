"""The scoping artifact: what was found, what was committed to, and what was refused.

The shape is the point. Discovery is unbounded and free; commitment is budgeted and loud.
Every item found lands in exactly ONE disposition and none may be dropped silently:

    committed  -- will be built, and carries an acceptance criterion
    non_goal   -- deliberately not built, and carries a reason
    assumption -- a decision the request did not make, and carries the default chosen

That invariant is what lets the expansion generators run wide without the scope running
wide with them: finding a thing costs nothing, committing to it is a separate act.

Identity and freshness are stored, not inferred. A previous incarnation of this idea keyed
its spec file on the session's launch directory, so a session started from $HOME inherited
a spec left there by an unrelated task four days earlier and ran a whole session against
another project's criteria. A pass therefore records the session it belongs to and the cwd
it was made in, and `stale_reason` refuses it when either stops matching.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flightdeck.models import DISPOSITIONS, ScopeRecord
from flightdeck.scope_gate import NONE, GateDecision
from flightdeck.store import DEFAULT_DIR, Store

#: Keyed by session, NOT by cwd. See the module docstring.
STATE_SUBDIR = "scope"

#: A pass older than this is presumed to belong to earlier work. The bleed that motivated
#: this was four days old and still being re-injected every turn.
MAX_AGE_SECONDS = 12 * 3600


@dataclass
class FoundItem:
    text: str
    disposition: str
    detail: str = ""          # acceptance criterion / reason / chosen default
    late: bool = False        # discovered after implementation began -- the coverage KPI
    source: str = "manual"    # which generator surfaced it

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"unknown disposition {self.disposition!r}; expected one of {DISPOSITIONS}"
            )


@dataclass
class Question:
    text: str
    changed_commitment: bool = False   # a question whose answer changed nothing was waste


@dataclass
class ScopePass:
    session_id: str
    cwd: str
    tier: str
    goal: str
    created_at: int = field(default_factory=lambda: int(time.time()))
    gate_reasons: list[str] = field(default_factory=list)
    found: list[FoundItem] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    divergence_flagged: int = 0
    implementation_started: bool = False
    implementation_started_at: int | None = None
    ceremony_tokens: int | None = None
    host: str = "local"

    #: The identity of the gate-log line this pass came from. Without it the Stop hook
    #: closes under its own id scheme while `scope_ingest` ingests the same decision under
    #: `record_id_for`, and one turn lands in scope_records TWICE -- same verdict, two rows,
    #: every rate computed over an inflated denominator. Carrying the gate's own seed makes
    #: both writers agree on the id, so the ingest skip-existing path dedupes them.
    gate_ts: str = ""
    prompt_sha1: str = ""

    # ---------------------------------------------------------------- construction

    @classmethod
    def open(
        cls, session_id: str, cwd: str, goal: str, decision: GateDecision, **kw: Any
    ) -> ScopePass:
        if decision.tier == NONE:
            raise ValueError("gate returned tier 'none'; do not open a pass -- that is the point")
        return cls(
            session_id=session_id, cwd=cwd, tier=decision.tier, goal=goal,
            gate_reasons=list(decision.reasons), **kw
        )

    # ---------------------------------------------------------------- mutation

    def add(
        self, text: str, disposition: str, detail: str = "", source: str = "manual"
    ) -> FoundItem:
        """Record a found item. `late` is set automatically once implementation has begun --
        the caller does not get to decide whether its own miss counts."""
        item = FoundItem(text=text, disposition=disposition, detail=detail,
                         late=self.implementation_started, source=source)
        self.found.append(item)
        return item

    def ask(self, text: str, *, changed_commitment: bool = False) -> None:
        self.questions.append(Question(text=text, changed_commitment=changed_commitment))

    def begin_implementation(self, *, now: int | None = None) -> None:
        """Marks the end of scoping and the start of building. The gap is the ceremony
        cost -- the figure that stops late_discovery_rate being improved by scoping
        forever, and the one the ledger tax showed is real."""
        self.implementation_started = True
        self.implementation_started_at = now or int(time.time())

    # ---------------------------------------------------------------- validation

    def problems(self) -> list[str]:
        """Everything wrong with this pass, all at once. Returning the first failure only
        would hide the rest behind a fix-and-rerun loop."""
        issues: list[str] = []
        if not self.goal.strip():
            issues.append("goal is empty")
        if not self.found:
            issues.append("no items found: a scoping pass that discovered nothing did not run")
        for item in self.found:
            if item.disposition == "committed" and not item.detail.strip():
                issues.append(f"committed without an acceptance criterion: {item.text[:60]!r}")
            if item.disposition == "non_goal" and not item.detail.strip():
                issues.append(f"non-goal without a reason: {item.text[:60]!r}")
            if item.disposition == "assumption" and not item.detail.strip():
                issues.append(f"assumption without a chosen default: {item.text[:60]!r}")
        if not any(i.disposition == "committed" for i in self.found):
            issues.append("nothing committed: the pass refused everything it found")
        return issues

    @property
    def valid(self) -> bool:
        return not self.problems()

    # ---------------------------------------------------------------- counts

    def counts(self) -> dict[str, int]:
        by = {d: 0 for d in DISPOSITIONS}
        for item in self.found:
            by[item.disposition] += 1
        return by

    # ---------------------------------------------------------------- persistence

    @staticmethod
    def state_dir(directory: Path | str = DEFAULT_DIR) -> Path:
        return Path(directory).expanduser() / STATE_SUBDIR

    def path(self, directory: Path | str = DEFAULT_DIR) -> Path:
        return self.state_dir(directory) / f"{self.session_id}.json"

    def save(self, directory: Path | str = DEFAULT_DIR) -> Path:
        target = self.path(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=1))
        return target

    @classmethod
    def load(cls, session_id: str, directory: Path | str = DEFAULT_DIR) -> ScopePass | None:
        target = cls.state_dir(directory) / f"{session_id}.json"
        if not target.exists():
            return None
        raw = json.loads(target.read_text())
        raw["found"] = [FoundItem(**i) for i in raw.get("found", [])]
        raw["questions"] = [Question(**q) for q in raw.get("questions", [])]
        return cls(**raw)

    # ---------------------------------------------------------------- freshness

    def stale_reason(
        self, cwd: str, *, now: int | None = None, max_age: int = MAX_AGE_SECONDS
    ) -> str | None:
        """Why this pass must not be reused, or None. Checked rather than assumed: the
        failure being guarded is a four-day-old spec from another project governing a
        session because both happened to start in the same directory."""
        age = (now or int(time.time())) - self.created_at
        if age > max_age:
            return f"scope is {age / 3600:.1f}h old (limit {max_age / 3600:.0f}h)"
        if cwd != self.cwd:
            return f"scope was made in {self.cwd}, this session is in {cwd}"
        return None

    def is_fresh_for(self, cwd: str, **kw: Any) -> bool:
        return self.stale_reason(cwd, **kw) is None

    # ---------------------------------------------------------------- emission

    def to_record(self, record_id: str, *, verdict: str | None = None) -> ScopeRecord:
        counts = self.counts()
        asked = len(self.questions)
        return ScopeRecord(
            record_id=record_id,
            session_id=self.session_id,
            created_at=self.created_at,
            host=self.host,
            cwd=self.cwd,
            tier=self.tier,
            verdict=verdict or ("pass" if self.valid else "fail"),
            found_total=len(self.found),
            committed=counts["committed"],
            non_goals=counts["non_goal"],
            assumptions=counts["assumption"],
            late_discovered=sum(1 for i in self.found if i.late),
            questions_asked=asked,
            questions_valuable=sum(1 for q in self.questions if q.changed_commitment),
            divergence_flagged=self.divergence_flagged,
            divergence_kept=sum(1 for i in self.found if i.source == "divergence"),
            ceremony_ms=(
                (self.implementation_started_at - self.created_at) * 1000
                if self.implementation_started_at is not None
                else None
            ),
            ceremony_tokens=self.ceremony_tokens,
            provenance={"gate_reasons": self.gate_reasons, "problems": self.problems()},
        )

    def close(self, store: Store, record_id: str, **kw: Any) -> ScopeRecord:
        record = self.to_record(record_id, **kw)
        store.add_scope_record(record)
        return record
