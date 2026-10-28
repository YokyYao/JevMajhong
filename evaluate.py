"""Evaluate all held-out decisions with explicit training-compatible token limits."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from jevany import JevModel
    from jevany.api import validate_distribution
    from jevany.inference import InferenceOptions
    model = JevModel.from_pretrained(args.checkpoint, device="cuda", dtype="bf16",
        inference_options=InferenceOptions(max_state_tokens=4096,
                                          max_branch_tokens=8192, max_packed_tokens=8192))
    results = []
    for line in args.data.read_text().splitlines():
        record = json.loads(line)
        questions = {name: {k: v for k, v in q.items() if k not in ("label", "target")}
                     for name, q in record["questions"].items()}
        response = model.system_one(state=record["state"], questions=questions)
        for name, question in questions.items():
            keys = list(question["criteria"])
            prob, _ = validate_distribution(response["answers"][name]["probabilities"], keys)
            label = record["questions"][name]["label"]
            results.append({"id": record["_meta"]["id"],
                            "correct": response["answers"][name]["choice"] == label,
                            "nll": -math.log(max(float(prob[keys.index(label)]), 1e-30))})
    if not results:
        raise ValueError("Empty evaluation")
    report = {"n": len(results), "accuracy": sum(x["correct"] for x in results)/len(results),
              "nll": sum(x["nll"] for x in results)/len(results), "rows": results,
              "note": "Tiny imitation validation, not playing strength; no temperature fitting."}
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
