"""Publish a dated, sanitized snapshot of completed evaluations to docs/."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Logged experiment directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs"
    run = args.run.resolve()
    metadata = json.loads((run / "metadata.json").read_text())
    evaluations = []
    for line in (run / "evaluations.jsonl").read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break  # A running experiment may still be writing the last record.
        # Explicit fields prevent unrelated future logger fields from being published.
        fields = ("evaluation_id", "iteration", "kind", "prompt", "objective",
                  "pass_objectives", "duration_seconds", "elapsed_seconds", "seeds",
                  "image_ids", "actions", "mean_action", "metrics", "best_so_far")
        evaluations.append({key: record[key] for key in fields if key in record})
    if not evaluations:
        raise SystemExit("No completed evaluations are available; existing snapshot left unchanged.")
    images = []
    for image in json.loads((run / "images.json").read_text()):
        source = (run / image["asset"]).resolve()
        if not source.is_relative_to(run):
            raise ValueError("Image asset must be inside the experiment directory")
        if image["split"] not in ("optimization", "heldout"):
            raise ValueError("Unknown image split")
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != image["sha256"]:
            raise ValueError(f"Image hash mismatch: {source.name}")
        target = destination / "assets/observations" / image["split"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        images.append({"split": image["split"], "id": image["id"], "sha256": actual_hash,
                       "asset": target.relative_to(destination).as_posix()})
    snapshot = {
        "schemaVersion": 1, "capturedAt": datetime.now(timezone.utc).isoformat(),
        "status": metadata["status"], "budget": metadata["budget"],
        "completedCandidates": sum(r["kind"] == "candidate" for r in evaluations),
        "model": metadata["model_id"], "modelRevision": metadata["model_revision"],
        "seed": metadata["seed"], "objective": metadata["objective"],
        "target_action": metadata["target_action"], "limitations": metadata["limitations"],
        "images": images, "evaluations": evaluations,
    }
    out = destination / "data/run-snapshot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    temporary.replace(out)
    print(json.dumps({"status": snapshot["status"], "candidates": snapshot["completedCandidates"],
                      "budget": snapshot["budget"], "capturedAt": snapshot["capturedAt"]}))


if __name__ == "__main__":
    main()
