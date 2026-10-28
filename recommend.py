"""Rank supplied legal actions from one exported decision observation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("request", type=Path, help="JSON request, or first record of a JSONL")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    from jevany import JevModel
    from jevany.api import validate_distribution
    from jevany.inference import InferenceOptions

    # Never pass labels/provenance to inference, even when a training row is supplied.
    with args.request.open() as stream:
        record = json.loads(stream.readline())
    questions = {name: {k: v for k, v in q.items() if k not in ("label", "target")}
                 for name, q in record["questions"].items()}
    model = JevModel.from_pretrained(args.checkpoint, device=args.device,
        dtype="bf16" if args.device == "cuda" else "fp32",
        inference_options=InferenceOptions(max_state_tokens=4096,
                                          max_branch_tokens=8192, max_packed_tokens=8192))
    response = model.system_one(state=record["state"], questions=questions)
    result = {}
    for name, question in questions.items():
        keys = list(question["criteria"])
        probabilities, _ = validate_distribution(response["answers"][name]["probabilities"], keys)
        candidates = sorted(zip(keys, probabilities), key=lambda x: -x[1])[:3]
        result[name] = [{"action": question["criteria"][key], "key": key,
                         "model_probability": float(prob)} for key, prob in candidates]
    print(json.dumps({"top3": result, "note": "Uncalibrated action scores, not win probabilities."}))


if __name__ == "__main__":
    main()
