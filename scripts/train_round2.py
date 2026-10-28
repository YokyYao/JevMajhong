"""Run the pinned trainer with matching validation limits and action diagnostics."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def action_summary(rows, targets):
    groups = defaultdict(list)
    for row in rows:
        if row["question"] != "action" or row["id"] not in targets:
            raise ValueError("Unexpected validation prediction")
        probabilities = row["p"]
        item = {"correct": max(range(len(probabilities)), key=probabilities.__getitem__) == row["label"],
                "nll": -math.log(max(probabilities[row["label"]], 1e-30))}
        groups[targets[row["id"]]].append(item)
    return {name: {"n": len(values), "accuracy": sum(v["correct"] for v in values)/len(values),
                   "nll": sum(v["nll"] for v in values)/len(values)} for name, values in sorted(groups.items())}


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    import torch
    import jevany.predictors as predictors
    import jevany.train as trainer
    from jevany.api import question_keys
    from jevany.data import materialize
    from jevany.device import sync
    from jevany.model import ContextLengthError

    dataset = ROOT / "data/semantic-2026-round2"
    audit = json.loads((dataset / "audit.json").read_text())
    if not audit["complete"] or audit["train_records"] != 64000 or audit["action_types"].get("dahai") != 54400:
        raise ValueError("Round-two dataset audit does not satisfy the frozen protocol")
    for filename, field in [("train.jsonl", "train_sha256"), ("validation.jsonl", "validation_sha256")]:
        if hashlib.sha256((dataset/filename).read_bytes()).hexdigest() != audit[field]:
            raise ValueError("Dataset changed after audit")
    validation = [json.loads(s) for s in (dataset / "validation.jsonl").read_text().splitlines()]
    if len(validation) != 1024 or any(r["_meta"]["split"] != "validation" for r in validation):
        raise ValueError("Expected the full, fixed 1024-record validation split")
    targets = {r["_meta"]["id"]: r["questions"]["action"]["criteria"][r["questions"]["action"]["label"]]["type"]
               for r in validation}
    if len(targets) != 1024:
        raise ValueError("Duplicate validation IDs")

    class ValidationPredictor(predictors.ModelPredictor):
        @torch.no_grad()
        def __call__(self, record):
            # Upstream benchmark defaults to a smaller frozen-suite window.
            # Keep this experiment's validation identical to training/recommendation limits.
            encoded = self.model.encode(self.tok, materialize(record), max_state=4096,
                                        max_branch=8192, strict=True)
            if len(encoded["ids"]) > 8192:
                raise ContextLengthError("Validation record exceeds 8192 packed tokens")
            sync(self.device)
            started = time.perf_counter()
            logits = self.model.forward(encoded)
            probabilities = [torch.softmax(z.float(), -1).cpu().tolist() for z in logits]
            values = [z.float().cpu().tolist() for z in logits]
            sync(self.device)
            keys = {name: question_keys(q["type"], q.get("criteria")) for name, q in record["questions"].items()}
            return {"probabilities": {name: dict(zip(keys[name], p)) for name, p in zip(keys, probabilities)},
                    "logits": {name: dict(zip(keys[name], z)) for name, z in zip(keys, values)},
                    "inference_temperature": self.temperature,
                    "latency_ms": 1000*(time.perf_counter()-started), "input_tokens": len(encoded["ids"])}

    original_evaluate = trainer.evaluate_during_training
    predictors.ModelPredictor = ValidationPredictor

    def evaluate(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            summary = original_evaluate(*args, **kwargs)
        run, step = Path(args[4]), args[5]
        directory = run / "training_eval" / f"step-{step:06d}"
        rows = json.loads((directory / "development/rows.json").read_text())
        if len(rows) != 1024 or len({r['id'] for r in rows}) != 1024 or summary["coverage"]["rejected_records"]:
            raise ValueError("Full validation coverage is required; cannot silently skip overlong rows")
        breakdown = action_summary(rows, targets)
        result = {"step": step, "n": 1024, "accuracy": summary["clean"]["acc"],
                  "nll": summary["clean"]["nll"], "by_target_action": breakdown}
        write(directory / "action-summary.json", result)
        curve = [json.loads(p.read_text()) for p in sorted((run / "training_eval").glob("step-*/action-summary.json"))]
        write(run / "validation-curve.json", curve)
        print("action_validation " + json.dumps(result), flush=True)
        return summary

    trainer.evaluate_during_training = evaluate
    trainer.main(["--config", str(ROOT / "semantic-round2.toml")])
    run = ROOT / "runs/semantic-2026-round2"
    curve = json.loads((run / "validation-curve.json").read_text())
    best = min(curve, key=lambda r: (r["nll"], r["step"]))
    checkpoint = run / "checkpoints" / f"step-{best['step']:06d}"
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError("The minimum-NLL checkpoint is incomplete")
    write(run / "selected-validation.json", {"step": best["step"], "nll": best["nll"],
                                            "checkpoint": str(checkpoint.relative_to(ROOT)),
                                            "selection_rule": "minimum full-validation NLL; ties prefer earlier step"})


if __name__ == "__main__":
    main()
