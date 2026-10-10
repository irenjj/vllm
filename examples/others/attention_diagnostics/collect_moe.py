# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a manifest of token-in/token-out snapshots and export MoE routes."""

import argparse
import copy
import json
import os
import urllib.request
from pathlib import Path


def collect(task, root, endpoint, headers, layers):
    replay = json.loads((root / task["replay_file"]).read_text())
    positions = task.get("query_positions", list(range(len(replay["token_ids"]))))
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("Select nonempty unique query positions")
    merged = None
    for start in range(0, len(positions), 128):
        selected = positions[start : start + 128]
        payload = copy.deepcopy(replay)
        payload.update(stream=False, output_mode="tokens")
        payload["sampling_params"] = {
            "attention_diagnostics": {
                "capture_kind": "moe",
                "query_positions": selected,
                "layer_names": layers,
            }
        }
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/inference/v1/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **headers},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)["choices"][0]["attention_diagnostics"]
        if "error" in data:
            raise RuntimeError(data["error"])
        if (
            data.get("capture_kind") != "moe"
            or data["query_positions"] != selected
            or data["sequence_length"] != len(replay["token_ids"])
            or set(data["layers"]) != set(layers)
        ):
            raise ValueError("Replay response does not match selected snapshot")
        if merged is None:
            merged = data
        else:
            merged["query_positions"].extend(selected)
            for name, entry in data["layers"].items():
                dest = merged["layers"][name]
                if entry["num_experts"] != dest["num_experts"]:
                    raise ValueError("Expert count changed between replay batches")
                for field in ("expert_ids", "routing_weights"):
                    dest[field].extend(entry[field])
        print(f"{task['id']}: {min(start + 128, len(positions))}/{len(positions)}")
    result = {key: task[key] for key in ("id", "name", "model")}
    result["diagnostics"] = merged
    if "tokens_file" in task:
        result["tokens"] = json.loads((root / task["tokens_file"]).read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique and nonempty")
    if len({t["model"] for t in tasks}) != 1:
        raise ValueError("Task group must use one model revision")
    headers = {}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    results = [
        collect(
            t, args.manifest.parent, args.endpoint, headers, manifest["layer_names"]
        )
        for t in tasks
    ]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema": "vllm-moe-diagnostics-v1", "tasks": results})
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
