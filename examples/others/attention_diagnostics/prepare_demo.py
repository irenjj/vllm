# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a real Qwen-style video replay; keep generated artifacts out of Git."""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import httpx
import pybase64 as base64
import regex as re
from transformers import AutoTokenizer

from vllm.entrypoints.scale_out.token_in_token_out.mm_features import (
    mm_kwargs_from_features,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import MultiModalFeatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--moe-layers", nargs="+", default=[])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument("--prompt", default="Describe the colors and motion briefly.")
    args = parser.parse_args()
    headers = {}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    client = httpx.Client(
        base_url=args.endpoint, headers=headers, timeout=600, trust_env=False
    )

    def post(path, payload):
        response = client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    video = base64.b64encode(args.video.read_bytes()).decode()
    request = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64," + video},
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "media_io_kwargs": {"video": {"fps": 2}},
        "mm_processor_kwargs": {"do_sample_frames": False},
    }
    replay = post("/v1/chat/completions/render", request)
    for key in ("request_id", "reasoning_parser_kwargs"):
        replay.pop(key, None)
    replay.update(
        model=args.model,
        stream=False,
        output_mode="tokens",
        sampling_params={"temperature": 0, "max_tokens": 32},
    )
    generated = post("/inference/v1/generate", replay)
    prompt_length = len(replay["token_ids"])
    replay["token_ids"] += generated["choices"][0]["token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    features = MultiModalFeatures.model_validate(replay["features"])
    items = mm_kwargs_from_features(features)["video"]
    if len(items) != 1:
        raise ValueError("The demo requires exactly one video")
    grid = items[0]["video_grid_thw"].data.flatten().tolist()
    nt, height, width = grid
    merge = args.spatial_merge_size
    if merge < 1 or height % merge or width % merge:
        raise ValueError("Invalid spatial merge size")
    grid = [nt, height // merge, width // merge]
    visual = [
        span.offset + i
        for span in features.mm_placeholders["video"]
        for i in range(span.length)
        if span.is_embed is None or span.is_embed[i]
    ]
    if len(visual) != grid[0] * grid[1] * grid[2]:
        raise ValueError("Visual token count does not match the processed video grid")
    visual_set = set(visual)
    tokens = [
        {"position": i, "text": tokenizer.decode([token]), "visual": i in visual_set}
        for i, token in enumerate(replay["token_ids"])
    ]
    timestamps = [
        float(t)
        for t in re.findall(
            r"<([0-9.]+) seconds>", "".join(t["text"] for t in tokens[:prompt_length])
        )
    ]
    if len(timestamps) != nt:
        raise ValueError("This processor does not expose one timestamp per video grid")
    queries = [
        t["position"]
        for t in tokens[prompt_length:]
        if t["text"].strip() and "<|" not in t["text"]
    ][:2]
    if not queries:
        raise ValueError("No generated text tokens available to demonstrate")
    payload = copy.deepcopy(replay)
    payload["sampling_params"] = {
        "attention_diagnostics": {
            "query_positions": queries,
            "layer_names": args.layers[:1],
        }
    }
    diagnostic = post("/inference/v1/generate", payload)["choices"][0][
        "attention_diagnostics"
    ]
    if "error" in diagnostic:
        raise ValueError(diagnostic["error"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "replay.json").write_text(json.dumps(replay))
    (args.output / "data.json").write_text(
        json.dumps(
            {
                "model_revision": args.model_revision,
                "layer_names": args.layers,
                "moe_layer_names": args.moe_layers,
                "prompt_length": prompt_length,
                "tokens": tokens,
                "grid": grid,
                "timestamps": timestamps,
                "visual_positions": visual,
                "diagnostics": diagnostic,
            },
            ensure_ascii=False,
        )
    )
    shutil.copyfile(args.video, args.output / "video.mp4")
    if args.moe_layers:
        for name in ("moe.html", "moe.js"):
            shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    for name in ("attention.html", "attention.js"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    print(
        f"Prepared {len(tokens)} tokens and {len(visual)} visual tokens "
        f"in {args.output}"
    )


if __name__ == "__main__":
    main()
