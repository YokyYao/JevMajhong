"""Verify the scheduler-assigned CUDA device has no other compute process."""
import csv
import io
import json
import os
import subprocess

import torch


def normalized_uuid(value):
    return str(value).lower().removeprefix("gpu-")


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This preflight requires exactly one scheduler-assigned GPU")
    device = torch.cuda.get_device_properties(0)
    actual_uuid = normalized_uuid(device.uuid)
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True)
    others = []
    for fields in csv.reader(io.StringIO(raw)):
        if len(fields) != 3 or normalized_uuid(fields[0].strip()) != actual_uuid:
            continue
        pid = int(fields[1].strip())
        if pid != os.getpid():
            others.append(fields[2].strip())
    print(json.dumps({"name": device.name, "uuid": str(device.uuid),
                      "cuda_mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "slurm_gpus": os.environ.get("SLURM_JOB_GPUS"),
                      "other_compute_processes": len(others),
                      "other_compute_memory_mib": others}), flush=True)
    if others:
        raise RuntimeError("Assigned GPU already has other compute processes; training was not started")


if __name__ == "__main__":
    main()
