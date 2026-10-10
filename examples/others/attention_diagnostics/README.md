# Token attention video demo

This local example calls the experimental token-in/token-out replay endpoint.
It keeps the original processed IDs and multimodal features, displays text
attention and overlays visual-token weights on the input video. It does not
capture the original decode run or provide causal attribution.

Start a supported Qwen-style VLM with the options documented in
`docs/features/attention_diagnostics.md`. Supply its exact attention module
names and the checkpoint's tokenizer. Authentication uses `VLLM_API_KEY`.

Create a two-second test video (requires FFmpeg), then prepare a real example:

```bash
ffmpeg -f lavfi -i testsrc2=size=320x240:rate=10 -t 2 -pix_fmt yuv420p /tmp/demo.mp4
uv run --no-project .venv/bin/python examples/others/attention_diagnostics/prepare_demo.py \
  --video /tmp/demo.mp4 --model your-served-model --tokenizer /path/to/checkpoint \
  --model-revision exact-weight-revision \
  --layers language_model.model.layers.7.self_attn.attn \
  --output /tmp/attention-demo
uv run --no-project .venv/bin/python examples/others/attention_diagnostics/serve.py \
  /tmp/attention-demo --port 8081
```

Open `http://localhost:8081/attention.html`. Select input/output tokens, click
replay, change heads locally, and play/scrub the video. Multiple token/head rows
are averaged. Selecting a new layer requires replay. The server is bound to
loopback by default and does not expose its API key to JavaScript.

`prepare_demo.py` supports a single video with `video_grid_thw`, a spatial merge
factor (default 2), placeholder embedding masks, and Qwen-style timestamp tokens.
Unsupported mappings fail explicitly. The video overlay uses the nearest
sampled time group; it does not invent attention for unsampled frames.

Generated `replay.json`, `data.json`, and `video.mp4` belong in an output directory,
not Git. They contain the original inputs/outputs and media. Retain them only
where that data is authorized. In-browser result caching is temporary.
