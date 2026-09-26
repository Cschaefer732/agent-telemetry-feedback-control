"""Argument wiring for `flightdeck synth ...`. No logic of its own -- everything here
delegates to a tested module, matching `__main__.py`'s `cmd_*(args) -> int` convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightdeck.store import DEFAULT_DIR, Store
from flightdeck.synth.manifest import Manifest, content_hash, verify_manifest, write_manifest
from flightdeck.synth.prompt_labels import (
    PromptLabelParams,
    generate_labels,
    seed_cases,
)
from flightdeck.synth.prompt_labels import (
    to_corpus as labels_to_corpus,
)
from flightdeck.synth.provenance import refuse_if_live_store
from flightdeck.synth.scope_rows import ScopeRowParams, generate_rows
from flightdeck.synth.scope_rows import to_corpus as rows_to_corpus


def _write_corpus(
    rows: list[dict], out: Path, *, kind: str, seed: int, params: dict, count: int
) -> None:
    out = Path(out)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    write_manifest(
        manifest_path,
        Manifest(kind=kind, seed=seed, params=params, count=count, content_hash=content_hash(rows)),
    )


def cmd_synth_rows(args: argparse.Namespace) -> int:
    params = ScopeRowParams(count=args.count, seed=args.seed, host=args.host)
    rows = rows_to_corpus(generate_rows(params))
    if args.dry_run:
        print(
            json.dumps(
                {"count": len(rows), "content_hash": content_hash(rows), "sample": rows[:3]},
                indent=2,
            )
        )
        return 0
    _write_corpus(
        rows,
        Path(args.out),
        kind="scope_rows",
        seed=args.seed,
        params=vars(params),
        count=len(rows),
    )
    return 0


def cmd_synth_labels(args: argparse.Namespace) -> int:
    if args.seed_cases:
        labels = labels_to_corpus(seed_cases())
        params: dict = {"kind": "seed_cases"}
    else:
        p = PromptLabelParams(count=args.count, seed=args.seed)
        labels = labels_to_corpus(generate_labels(p))
        params = vars(p)
    if args.dry_run:
        print(
            json.dumps(
                {"count": len(labels), "content_hash": content_hash(labels), "sample": labels[:3]},
                indent=2,
            )
        )
        return 0
    _write_corpus(
        labels,
        Path(args.out),
        kind="prompt_labels",
        seed=args.seed,
        params=params,
        count=len(labels),
    )
    return 0


def cmd_synth_validate(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    ok = verify_manifest(Path(args.manifest), rows)
    print(json.dumps({"valid": ok}))
    return 0 if ok else 1


def cmd_synth_load(args: argparse.Namespace) -> int:
    from flightdeck.models import ScopeRecord

    refuse_if_live_store(args.dir, DEFAULT_DIR, what="synthetic scope_records")
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    with Store(args.dir) as store:
        for row in rows:
            correction_families = row.get("correction_families")
            if isinstance(correction_families, str):
                correction_families = json.loads(correction_families) if correction_families else {}
            provenance = row.get("provenance")
            if isinstance(provenance, str):
                provenance = json.loads(provenance) if provenance else {}
            known = {f.name for f in ScopeRecord.__dataclass_fields__.values()}
            record_kwargs = {
                k: v
                for k, v in row.items()
                if k in known and k not in ("correction_families", "provenance")
            }
            record = ScopeRecord(
                correction_families=correction_families or {},
                provenance=provenance or {},
                **record_kwargs,
            )
            store.add_scope_record(record)
    print(json.dumps({"loaded": len(rows)}))
    return 0


def cmd_synth_dry_run(args: argparse.Namespace) -> int:
    if args.kind == "rows":
        params = ScopeRowParams(count=args.count, seed=args.seed)
        rows = rows_to_corpus(generate_rows(params))
    else:
        params = PromptLabelParams(count=args.count, seed=args.seed)
        rows = labels_to_corpus(generate_labels(params))
    print(
        json.dumps(
            {"count": len(rows), "content_hash": content_hash(rows), "sample": rows[:3]}, indent=2
        )
    )
    return 0


def build_subparser(subparsers: argparse._SubParsersAction) -> None:
    synth = subparsers.add_parser("synth", help="synthetic data generation")
    synth_sub = synth.add_subparsers(dest="synth_cmd", required=True)

    rows_p = synth_sub.add_parser("rows")
    rows_p.add_argument("--count", type=int, default=60)
    rows_p.add_argument("--seed", type=int, default=20260905)
    rows_p.add_argument("--host", default="synth")
    rows_p.add_argument("--out")
    rows_p.add_argument("--dry-run", action="store_true")
    rows_p.set_defaults(func=cmd_synth_rows)

    labels_p = synth_sub.add_parser("labels")
    labels_p.add_argument("--count", type=int, default=60)
    labels_p.add_argument("--seed", type=int, default=20260905)
    labels_p.add_argument(
        "--seed-cases", action="store_true", help="emit the balanced seed-case corpus"
    )
    labels_p.add_argument("--out")
    labels_p.add_argument("--dry-run", action="store_true")
    labels_p.set_defaults(func=cmd_synth_labels)

    validate_p = synth_sub.add_parser("validate")
    validate_p.add_argument("--manifest", required=True)
    validate_p.add_argument("--data", required=True)
    validate_p.set_defaults(func=cmd_synth_validate)

    load_p = synth_sub.add_parser("load")
    load_p.add_argument("--data", required=True)
    load_p.add_argument("--dir", required=True)
    load_p.set_defaults(func=cmd_synth_load)

    dry_p = synth_sub.add_parser("dry-run")
    dry_p.add_argument("--count", type=int, default=60)
    dry_p.add_argument("--seed", type=int, default=20260905)
    dry_p.add_argument("--kind", choices=("rows", "labels"), default="rows")
    dry_p.set_defaults(func=cmd_synth_dry_run)


def dispatch(args: argparse.Namespace) -> int:
    return args.func(args)
