from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.analyze_moving_objects import analyze, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    job_ids = [row["job_id"] for row in rows]
    states = _job_states(job_ids)
    summary = analyze(args.run_root, {row["run_name"] for row in rows})
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "jobs": len(job_ids),
        "states": dict(Counter(states.get(job_id, "UNKNOWN") for job_id in job_ids)),
        "complete_runs": len(summary["runs"]),
        "groups": len(summary["aggregates"]),
    }
    output_root = args.manifest.parent / f"{args.manifest.stem}_oversight"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_root / "summary.md").write_text(render_markdown(summary))
    print(json.dumps(payload, sort_keys=True))


def _job_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", ",".join(job_ids), "-o", "JobIDRaw,State", "-P"],
        check=True, capture_output=True, text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2:
            states[fields[0]] = fields[1].split()[0]
    return states


if __name__ == "__main__":
    main()
