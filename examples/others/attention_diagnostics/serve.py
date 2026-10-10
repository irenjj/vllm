# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve a local attention replay demo prepared by prepare_demo.py."""

import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    root = args.directory.resolve()
    data = json.loads((root / "data.json").read_text())
    headers = {}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + key

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/video.mp4":
                return super().do_GET()
            path = root / "video.mp4"
            size = path.stat().st_size
            start, end = 0, size - 1
            partial = "Range" in self.headers
            if partial:
                try:
                    unit, value = self.headers["Range"].split("=", 1)
                    first, last = value.split("-", 1)
                    if unit != "bytes":
                        raise ValueError
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                    if not 0 <= start <= end < size:
                        raise ValueError
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = source.read(min(remaining, 65536))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_POST(self):
            if self.path != "/api/replay":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size")
                selection = json.loads(self.rfile.read(size))
                positions = selection["positions"]
                if (
                    not isinstance(positions, list)
                    or not 0 < len(positions) <= 128
                    or any(
                        type(i) is not int or not 0 <= i < len(data["tokens"])
                        for i in positions
                    )
                    or len(set(positions)) != len(positions)
                ):
                    raise ValueError("Invalid token positions")
                kind = selection.get("kind", "attention")
                if kind == "moe":
                    names = selection["layer_names"]
                    allowed = data.get("moe_layer_names", [])
                    if (
                        not isinstance(names, list)
                        or not 0 < len(names) <= 128
                        or any(n not in allowed for n in names)
                        or len(set(names)) != len(names)
                    ):
                        raise ValueError("Unknown MoE layers")
                elif kind == "attention":
                    names = [selection["layer_name"]]
                    if names[0] not in data["layer_names"]:
                        raise ValueError("Unknown attention layer")
                else:
                    raise ValueError("Unknown diagnostic kind")
                payload = json.loads((root / "replay.json").read_text())
                payload["sampling_params"] = {
                    "attention_diagnostics": {
                        "query_positions": positions,
                        "layer_names": names,
                        "capture_kind": kind,
                    }
                }
                response = httpx.post(
                    args.endpoint.rstrip("/") + "/inference/v1/generate",
                    json=payload,
                    headers=headers,
                    timeout=600,
                    trust_env=False,
                )
                response.raise_for_status()
                result = response.json()["choices"][0]["attention_diagnostics"]
                if "error" in result:
                    raise ValueError(result["error"])
                status = 200
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                result, status = {"error": str(exc)}, 400
            encoded = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    print(f"Demo: http://{args.host}:{args.port}/attention.html", flush=True)
    ThreadingHTTPServer(
        (args.host, args.port), partial(Handler, directory=str(root))
    ).serve_forever()


if __name__ == "__main__":
    main()
