"""Store schema and forward-only migrations.

The schema is the contract between every component. The Go emitter, the Claude Code hooks, the
scorer, the governor, and the nightly reviewer share nothing but this — no imports, no RPC. A
source that cannot observe a column writes NULL rather than being excluded from the table.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 16

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE turns (
            turn_id            TEXT PRIMARY KEY,
            session_id         TEXT NOT NULL,
            parent_session_id  TEXT,
            source             TEXT NOT NULL,
            host               TEXT NOT NULL,
            cwd                TEXT,
            git_sha            TEXT,
            turn_idx           INTEGER,
            started_at         INTEGER NOT NULL,
            ended_at           INTEGER,
            wall_ms            INTEGER,
            agent_name         TEXT,
            is_subagent        INTEGER NOT NULL DEFAULT 0,
            mode               TEXT,
            provider           TEXT,
            model              TEXT,
            tier               TEXT,
            model_ms           INTEGER,
            requests           INTEGER,
            retries            INTEGER,
            prompt_tokens      INTEGER,
            completion_tokens  INTEGER,
            cached_tokens      INTEGER,
            estimated          INTEGER NOT NULL DEFAULT 0,
            context_peak       INTEGER,
            context_window     INTEGER,
            outcome            TEXT,
            finish_reason      TEXT,
            error_class        TEXT,
            kpi_score          REAL,
            kpi_components     TEXT,
            flagged            INTEGER NOT NULL DEFAULT 0,
            judged             INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE events (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id     TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            kind        TEXT NOT NULL,
            name        TEXT,
            duration_ms INTEGER,
            ok          INTEGER,
            payload     TEXT
        );

        CREATE TABLE texts (
            turn_id    TEXT NOT NULL,
            kind       TEXT NOT NULL,
            seq        INTEGER NOT NULL,
            body       TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            PRIMARY KEY (turn_id, kind, seq)
        );

        CREATE TABLE judgments (
            turn_id     TEXT PRIMARY KEY,
            judge_model TEXT,
            verdict     TEXT,
            rubric      TEXT,
            notes       TEXT,
            lesson      TEXT,
            created_at  INTEGER NOT NULL
        );

        CREATE TABLE governor_decisions (
            turn_id         TEXT NOT NULL,
            domain          TEXT NOT NULL,
            chosen          TEXT NOT NULL,
            alternatives    TEXT,
            features        TEXT,
            shadow          INTEGER NOT NULL,
            weights_version TEXT,
            PRIMARY KEY (turn_id, domain)
        );

        CREATE TABLE probes (
            probe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       INTEGER NOT NULL,
            host     TEXT NOT NULL,
            kind     TEXT NOT NULL,
            ok       INTEGER,
            total    INTEGER,
            detail   TEXT
        );

        CREATE INDEX idx_turns_started ON turns(started_at);
        CREATE INDEX idx_turns_kpi ON turns(kpi_score);
        CREATE INDEX idx_turns_triage ON turns(flagged, judged);
        CREATE INDEX idx_turns_session ON turns(session_id, turn_idx);
        CREATE INDEX idx_events_turn ON events(turn_id, kind);
        CREATE INDEX idx_texts_expiry ON texts(expires_at);
        CREATE INDEX idx_probes_ts ON probes(ts, kind);
        """,
    ),
    (
        2,
        """
        -- Dedupe key for the canonical merge on spark. Two boxes can produce the same event
        -- row after a partial sync; without this the merge double-counts tool calls.
        CREATE UNIQUE INDEX idx_events_dedupe ON events(turn_id, ts, kind, IFNULL(name, ''));
        """,
    ),
    (
        3,
        """
        -- Written by the nightly reviewer, read by the next night's reviewer. A change that was
        -- reverted for regressing KPI must not be re-derived from the same evidence.
        CREATE TABLE tuning_changes (
            change_id     TEXT PRIMARY KEY,
            applied_at    INTEGER NOT NULL,
            domain        TEXT NOT NULL,
            path          TEXT NOT NULL,
            summary       TEXT NOT NULL,
            commit_sha    TEXT,
            evidence      TEXT,
            kpi_before    REAL,
            kpi_after     REAL,
            reverted_at   INTEGER,
            revert_reason TEXT
        );

        CREATE INDEX idx_tuning_applied ON tuning_changes(applied_at);
        """,
    ),
    (
        4,
        """
        -- Prefill and prompt-cache signals, promoted from event payloads to columns because the
        -- governor scores on them per turn. ttft_ms separates prefill from generation: on a local
        -- model a warm KV prefix returns a small TTFT even when total latency is long, so tier
        -- choice can account for cache warmth instead of only raw speed.
        ALTER TABLE turns ADD COLUMN ttft_ms INTEGER;

        -- How many times the active tool set changed mid-turn. Each change invalidates the prompt
        -- prefix, so this is what makes tool-search churn a measured cost rather than a suspicion.
        ALTER TABLE turns ADD COLUMN tools_hash_changes INTEGER;

        CREATE INDEX idx_turns_ttft ON turns(ttft_ms);
        """,
    ),
    (
        5,
        """
        -- Point-in-time CPU/RAM/GPU/disk snapshot per host, feeding fleet-status.md. GPU columns
        -- are NULL on hosts with no GPU (e.g. laptop). UNIQUE(host, ts) is the dedupe key
        -- merge_from's INSERT OR IGNORE relies on, same pattern as idx_events_dedupe.
        CREATE TABLE host_metrics (
            host            TEXT NOT NULL,
            ts              TEXT NOT NULL,
            cpu_pct         REAL,
            mem_used_mb     INTEGER,
            mem_total_mb    INTEGER,
            gpu_pct         REAL,
            gpu_mem_used_mb INTEGER,
            disk_used_gb    REAL,
            disk_total_gb   REAL,
            UNIQUE(host, ts)
        );

        CREATE INDEX idx_host_metrics_host_ts ON host_metrics(host, ts);
        """,
    ),
    (
        6,
        """
        -- governor_decisions' PK is (turn_id, domain), domain-last, so Governor.success_table's
        -- per-domain lookup (now a single join against turns instead of an N+1 scan of every
        -- turn ever recorded) couldn't use an index for its `domain = ?` filter. Index-only,
        -- no table change.
        CREATE INDEX idx_governor_decisions_domain ON governor_decisions(domain, turn_id);
        """,
    ),
    (
        7,
        """
        -- add_probe used to INSERT unconditionally, and cmd_rollup replayed the entire JSONL
        -- history every run, so every rollup re-inserted every probe ever written — exact
        -- duplicate rows, since replay reuses the JSONL record's original ts. Collapse any
        -- duplicates that bug already produced (same host/kind/ts, replayed from the same
        -- record) before enforcing the constraint that stops new ones.
        DELETE FROM probes
        WHERE probe_id NOT IN (SELECT MIN(probe_id) FROM probes GROUP BY host, kind, ts);

        CREATE UNIQUE INDEX idx_probes_dedupe ON probes(host, kind, ts);
        """,
    ),
    (
        8,
        """
        -- Per-file replay cursor. cmd_rollup used to call ingest_jsonl on every events-*.jsonl
        -- file on every run, replaying the entire history each time (O(total history) per
        -- rollup, and the direct cause of the probes duplication idx_probes_dedupe guards
        -- against). This table lets ingest_jsonl resume from the byte offset it left off at.
        CREATE TABLE ingest_cursor (
            path   TEXT PRIMARY KEY,
            offset INTEGER NOT NULL
        );
        """,
    ),
    (
        9,
        """
        -- idx_events_dedupe's key was (turn_id, ts, kind, IFNULL(name, '')) — two genuinely
        -- distinct tool_call events in the same turn that complete in the same millisecond
        -- collided on that key, and INSERT OR IGNORE silently dropped the second one, quietly
        -- undercounting tool_reliability. dedupe_id carries the event's tool_use_id (the one
        -- payload field a source that can distinguish concurrent calls actually supplies; empty
        -- string when a source can't, same collision-prone behavior those sources already had).
        -- It's derived fresh from payload at insert time (Store.add_events), including on JSONL
        -- replay, so a truly identical replayed row still computes the same dedupe_id and still
        -- collides — the merge-dedup property idx_events_dedupe exists for is preserved.
        ALTER TABLE events ADD COLUMN dedupe_id TEXT NOT NULL DEFAULT '';

        DROP INDEX idx_events_dedupe;
        CREATE UNIQUE INDEX idx_events_dedupe
            ON events(turn_id, ts, kind, IFNULL(name, ''), dedupe_id);
        """,
    ),
    (
        10,
        """
        -- Store.percentile had no index matching the filter its only caller (trailing_baseline)
        -- actually issues: source + tier + a started_at range. Every KPI scoring call during
        -- rollup did a full scan of `turns` to rank a handful of trailing-window candidates.
        -- Index-only, no table change. Sorting by the requested column (wall_ms, kpi_score, a
        -- token count, ...) still happens in a temp b-tree, but over the now-bounded filtered
        -- set instead of the whole table.
        CREATE INDEX idx_turns_percentile ON turns(source, tier, started_at);
        """,
    ),
    (
        11,
        """
        -- Human and model verdicts on a turn, side by side.
        --
        -- `judgments` could not hold this: its primary key is turn_id alone and its only
        -- author column is judge_model, so a human verdict would OVERWRITE a model's rather
        -- than sit beside it -- and comparing the two is the entire point of collecting
        -- human labels. It is also empty (0 rows), so nothing is lost by leaving it be.
        --
        -- The ledger snapshot is denormalised on purpose. Entry states mutate (active ->
        -- dormant -> evicted), so the entries table can only say what an entry is NOW, never
        -- what it was when the turn ran. A label whose ledger context has to be reconstructed
        -- later cannot tune the ledger, which is the subsystem it is most needed for.
        CREATE TABLE labels (
            turn_id         TEXT    NOT NULL,
            labeler         TEXT    NOT NULL,   -- 'human:<name>' or 'model:<id>'
            verdict         TEXT    NOT NULL,   -- good | bad | mixed
            -- Subsystem-scoped, e.g. 'ledger:dropped-needed', 'verify:claimed-untested'.
            -- A bare "bad turn" cannot tune anything; a tag names which subsystem to move.
            tags            TEXT    NOT NULL DEFAULT '[]',
            note            TEXT    NOT NULL DEFAULT '',
            ledger_carried  INTEGER,
            ledger_dropped  INTEGER,
            ledger_tokens   INTEGER,
            ledger_states   TEXT,               -- JSON {state: count} over carried+dropped
            created_at      INTEGER NOT NULL,
            PRIMARY KEY (turn_id, labeler)
        );
        CREATE INDEX idx_labels_verdict ON labels(verdict, created_at);
        CREATE INDEX idx_labels_labeler ON labels(labeler, created_at);
        """,
    ),
    (
        12,
        """
        -- One row per scoping pass. Separate from `turns` because a scoping pass spans
        -- several turns (and sometimes zero -- tier 'none' still records that the gate
        -- fired and chose not to ceremonialise, which is the measurement that catches
        -- ceremony creep).
        --
        -- verdict is three-valued (pass|fail|silent). A missing writer must read SILENT,
        -- never PASS: the previous capture layer sat wired to nothing while every check
        -- stayed green, and a binary column is what made that invisible.
        --
        -- Quality counters are stored beside their cost counters on purpose. late_discovered
        -- alone is gameable by scoping forever; ceremony_tokens alone is gameable by not
        -- scoping at all. The pair is the metric.
        CREATE TABLE scope_records (
            record_id             TEXT PRIMARY KEY,
            session_id            TEXT NOT NULL,
            turn_id               TEXT,
            created_at            INTEGER NOT NULL,
            host                  TEXT NOT NULL,
            cwd                   TEXT,
            git_sha               TEXT,
            tier                  TEXT NOT NULL,      -- none | mini | full
            verdict               TEXT NOT NULL,      -- pass | fail | silent
            found_total           INTEGER NOT NULL DEFAULT 0,
            committed             INTEGER NOT NULL DEFAULT 0,
            non_goals             INTEGER NOT NULL DEFAULT 0,
            assumptions           INTEGER NOT NULL DEFAULT 0,
            late_discovered       INTEGER NOT NULL DEFAULT 0,
            ceremony_ms           INTEGER,
            ceremony_tokens       INTEGER,
            questions_asked       INTEGER NOT NULL DEFAULT 0,
            questions_valuable    INTEGER NOT NULL DEFAULT 0,
            files_edited          INTEGER NOT NULL DEFAULT 0,
            rework_files          INTEGER NOT NULL DEFAULT 0,
            turns_to_first_edit   INTEGER,
            turns_to_done         INTEGER,
            corrections           INTEGER NOT NULL DEFAULT 0,
            correction_families   TEXT NOT NULL DEFAULT '{}',
            assumptions_overridden INTEGER NOT NULL DEFAULT 0,
            divergence_flagged    INTEGER NOT NULL DEFAULT 0,
            divergence_kept       INTEGER NOT NULL DEFAULT 0,
            provenance            TEXT NOT NULL DEFAULT '{}',
            prev_hash             TEXT,
            row_hash              TEXT
        );
        CREATE INDEX idx_scope_session ON scope_records(session_id, created_at);
        CREATE INDEX idx_scope_verdict ON scope_records(verdict, created_at);
        CREATE INDEX idx_scope_tier ON scope_records(tier, created_at);
        """,
    ),
    (
        13,
        """
        -- The active ledger: what is being worked on RIGHT NOW, by whom, where.
        --
        -- Heartbeat, not registration. A session that is killed, crashes, or has its
        -- machine sleep never gets to deregister, so presence is derived from
        -- `heartbeat_at` recency rather than from a status column a dead process was
        -- supposed to update. `status` records intent ('working', 'blocked', 'done');
        -- liveness is time. Reading status as liveness is how a registry fills with
        -- ghosts that all claim to be running.
        --
        -- Keyed by (session_id, host): the same session id can legitimately appear on
        -- two machines in this fleet, and a bare session_id primary key would let one
        -- machine's heartbeat silently overwrite the other's goal.
        CREATE TABLE agent_sessions (
            session_id      TEXT NOT NULL,
            host            TEXT NOT NULL,
            cwd             TEXT,
            repo            TEXT,            -- git toplevel, NULL outside a repo
            github_remote   TEXT,            -- owner/name, NULL when there is no remote
            branch          TEXT,
            goal            TEXT,            -- why this session exists, one line
            task            TEXT,            -- what it is doing at this moment
            agent           TEXT,            -- claude-code | sparky | subagent | ...
            status          TEXT NOT NULL DEFAULT 'working',
            started_at      INTEGER NOT NULL,
            heartbeat_at    INTEGER NOT NULL,
            PRIMARY KEY (session_id, host)
        );
        CREATE INDEX idx_agent_sessions_beat ON agent_sessions(heartbeat_at DESC);

        -- Indexes over things that live on disk. Every one stores the source path and a
        -- content hash: an index keyed on mtime silently goes stale across a checkout,
        -- which this project has already been bitten by once.
        CREATE TABLE memory_index (
            name            TEXT NOT NULL,
            project         TEXT NOT NULL,   -- the ~/.claude/projects/<slug> it belongs to
            path            TEXT NOT NULL,
            kind            TEXT,            -- user | feedback | project | reference
            description     TEXT,
            links           TEXT,            -- JSON array of [[wikilink]] targets
            content_sha     TEXT NOT NULL,
            indexed_at      INTEGER NOT NULL,
            PRIMARY KEY (project, name)
        );

        CREATE TABLE skill_index (
            name            TEXT PRIMARY KEY,
            path            TEXT NOT NULL,
            scope           TEXT,            -- user | project | plugin
            description     TEXT,
            trigger         TEXT,
            content_sha     TEXT NOT NULL,
            indexed_at      INTEGER NOT NULL
        );

        CREATE TABLE repo_index (
            path            TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            github_remote   TEXT,
            branch          TEXT,
            head_sha        TEXT,
            dirty           INTEGER NOT NULL DEFAULT 0,
            ahead           INTEGER,
            behind          INTEGER,
            last_commit_at  INTEGER,
            last_commit_subject TEXT,
            indexed_at      INTEGER NOT NULL
        );

        -- TODOs, local (scoped to a repo) and global (repo IS NULL). Deliberately not a
        -- second source of truth for the agent's in-flight TodoWrite list: this is the
        -- durable, cross-session layer that outlives one turn.
        CREATE TABLE todos (
            todo_id         TEXT PRIMARY KEY,
            scope           TEXT NOT NULL,   -- 'local' | 'global'
            repo            TEXT,            -- NULL for global
            text            TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'open',
            source          TEXT,            -- who filed it: session id, 'human', ...
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL,
            done_at         INTEGER
        );
        CREATE INDEX idx_todos_scope ON todos(scope, status);
        """,
    ),
    (
        14,
        """
        -- Every scheduler on this fleet, in one place. There are three (launchd on the
        -- Macs, systemd user timers on spark, crontab) and no surface previously answered
        -- "what is scheduled". The motivating find: this laptop's crontab holds five
        -- comments describing five jobs and ZERO job lines. A scheduler that fires nothing
        -- reads exactly like a scheduler with nothing to do.
        CREATE TABLE scheduled_jobs (
            job_id          TEXT PRIMARY KEY,   -- source|host|name
            source          TEXT NOT NULL,      -- launchd | systemd | cron
            host            TEXT NOT NULL,
            name            TEXT NOT NULL,
            schedule        TEXT,               -- as the source states it, not normalised
            command         TEXT,
            enabled         INTEGER,
            last_run_at     INTEGER,
            next_run_at     INTEGER,
            last_exit       INTEGER,
            status          TEXT,               -- running | waiting | failed | unknown
            note            TEXT,
            indexed_at      INTEGER NOT NULL
        );
        CREATE INDEX idx_scheduled_jobs_source ON scheduled_jobs(source, host);

        -- Reachability is stored SEPARATELY from the jobs, because "this source reported
        -- zero jobs" and "this source could not be reached" are different facts that a
        -- job-count alone collapses into the same number. A remote host timed out on the
        -- first run of this indexer; recording that as 0 jobs would have been a lie.
        CREATE TABLE schedule_sources (
            source          TEXT NOT NULL,
            host            TEXT NOT NULL,
            reachable       INTEGER NOT NULL,
            jobs_found      INTEGER,            -- NULL when unreachable, never 0
            error           TEXT,
            checked_at      INTEGER NOT NULL,
            PRIMARY KEY (source, host)
        );
        """,
    ),
    (
        15,
        """
        -- Propensity, so a logged decision can later be evaluated off-policy.
        --
        -- Governor._choose already computes whether a pick was the argmax or an epsilon-greedy
        -- exploration, and record() dropped both on the floor. Without them P(a|x) is
        -- unrecoverable, and every off-policy estimator (IPS, SNIPS, doubly-robust) is undefined
        -- on these rows -- not merely imprecise. Storing epsilon alongside the flag rather than
        -- reading it back from governor.toml is deliberate: the toml is hand-edited between runs,
        -- so the value in force at decision time is not recoverable from the file afterwards.
        --
        -- Existing rows keep explored=0/reason='' honestly: every decision recorded before this
        -- migration was made under shadow=1, where no arm was ever acted on. epsilon stays NULL
        -- for them because the value then in force was not written down.
        ALTER TABLE governor_decisions ADD COLUMN explored INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE governor_decisions ADD COLUMN reason TEXT NOT NULL DEFAULT '';
        ALTER TABLE governor_decisions ADD COLUMN epsilon REAL;
        """,
    ),
    (
        16,
        """
        -- Lets a producer file a todo idempotently: a probe that re-files the same failure
        -- every session start must not create duplicate rows. The index is partial (only rows
        -- that opt in with a non-NULL key are constrained), so every existing todo -- and every
        -- future ad-hoc one filed without a key -- is untouched.
        ALTER TABLE todos ADD COLUMN key TEXT;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_todos_source_key
            ON todos(source, key) WHERE key IS NOT NULL;

        -- The plan IS a todo (decided 2026-09-11, same shape as the todos-vs-Vikunja call in
        -- migration 13): one plan = one todos row, `goal` as its text. Stages are fields of
        -- the plan here, never separate todo rows -- a plan with four stages must not turn
        -- into five entries in `ledger show`. scope/repo live on the todo row only, so a plan
        -- is filterable the same way any other todo is.
        --
        -- Body fields are JSON-or-null on purpose: assumptions/alternatives/risks/gotchas/
        -- rollback/observability/open_questions/reasoning are free-form per plan (a one-stage
        -- fix has no alternatives worth recording; a migration has several), and forcing them
        -- into columns would mean a schema migration every time a new plan shape needs one
        -- more field. `stages` and `non_goals` are NOT NULL because a plan with no stages or
        -- no stated non-goals did not go through scoping -- same discipline as
        -- ScopePass.problems()'s "no items found" check.
        CREATE TABLE plans (
            todo_id         TEXT PRIMARY KEY REFERENCES todos(todo_id),
            goal            TEXT NOT NULL,
            stages          TEXT NOT NULL,
            non_goals       TEXT NOT NULL,
            assumptions     TEXT,
            alternatives    TEXT,
            risks           TEXT,
            gotchas         TEXT,
            rollback        TEXT,
            observability   TEXT,
            open_questions  TEXT,
            reasoning       TEXT,
            status          TEXT NOT NULL DEFAULT 'draft',
            superseded_by   TEXT,
            revision        INTEGER NOT NULL DEFAULT 1,
            session_id      TEXT,
            host            TEXT,
            created_at      INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
        CREATE INDEX idx_plans_status ON plans(status);

        -- Append-only. amend_plan/set_plan_status are compare-and-swap on plans.revision --
        -- never a lock -- and every accepted change writes one full snapshot here, so a
        -- conflicting amender's lost update is still reconstructable from history instead of
        -- silently overwritten with no trace.
        CREATE TABLE plan_revisions (
            todo_id     TEXT NOT NULL,
            revision    INTEGER NOT NULL,
            snapshot    TEXT NOT NULL,
            changed_by  TEXT,
            changed_at  INTEGER NOT NULL,
            note        TEXT,
            PRIMARY KEY (todo_id, revision)
        );
        """,
    ),
]


def migrate(conn: sqlite3.Connection) -> int:
    """Apply outstanding migrations. Returns the resulting schema version."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    current = row[0] or 0
    for version, sql in MIGRATIONS:
        if version <= current:
            continue
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        current = version
    conn.commit()
    return current
