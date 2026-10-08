# Attention diagnostic replay (experimental)

This extension reuses the model already loaded in the MRV2 GPU worker. It runs
one prefill over supplied token IDs, captures selected layers' post-RoPE Q/K,
and returns selected causal attention rows. It never samples a new token.
It is an inspection tool, not causal attribution or a guarantee of reproducing
bit-identical attention from a previous decode run.

## Server requirements

Use this checkout with `VLLM_USE_V2_MODEL_RUNNER=1`, `--enforce-eager`, and
`--no-async-scheduling`. The first implementation supports standard causal
MHA/GQA attention and tensor parallel head gathering. Set PP/DP/DCP/PCP to 1.
Speculative decoding, microbatching/DBO, expert parallelism, auxiliary output, cache transfer,
quantized KV, fast KV sharing, encoder-decoder and diffusion models are rejected.

Selected layers must use `vllm.model_executor.layers.attention.Attention` with
ordinary causal semantics. Sliding-window, ALiBi, logit softcap, attention sinks,
chunk-lookback, dual-chunk, KV-sharing and multimodal prefix attention are unsupported.
Selecting a linear-attention or MLA layer returns an explicit capture error.
A hybrid model can expose its standard full-attention layers; this does not
visualize its recurrent/linear layers.

A diagnostic request runs exclusively on its engine: existing generation work
finishes before it is admitted, and subsequent requests wait. This avoids mixing
prefill-only completion and normal sampling. Large diagnostics therefore affect
serving latency. Prefix-cache reads and writes are disabled for the diagnostic request so
all requested Q/K are actually recomputed. Normal requests keep their existing
sampling and cache behavior. No second weight copy is loaded.

## Early exit

By default (`early_exit=true`), the Llama and Qwen3Next decoder loops (including
Qwen3.5 models that inherit that loop) return immediately after the deepest
selected decoder block. Earlier blocks, including recurrent/linear blocks, still
run normally for every prefill chunk. The final norm and later blocks are skipped;
no logits or generated tokens are required. The response's `early_exit_layer`
reports the zero-based stopping layer, or null when the full loop ran.

Set `early_exit=false` for a full-forward numerical comparison. Other model loops
fall back to full forward. Missing layer names also fall back and return the usual
capture error. Selecting the last layer runs the full loop. Diagnostic blocks are
never published into the shared prefix cache, even with early exit disabled;
request-local KV/recurrent state remains available between prefill chunks.
This does not reduce cache allocation or skip the vision encoder.

## Token and layer selection

The engine receives one logical input sequence. To inspect a previous answer,
concatenate the original **actual** `prompt_token_ids` and generated `token_ids`.
Do not re-tokenize rendered answer text or apply the chat template again.
There is no request/response segment distinction in the diagnostic protocol.

Positions are zero-based in the final processed sequence, including special
and expanded multimodal tokens. They are sequence offsets, not RoPE/mRoPE
coordinates. The row at position `i` describes the token at `i` attending to
positions `0..i`; it is not automatically shifted to the state that predicted
that token. Multiple positions can be selected in any order.

`layer_names` are exact attention prefixes supplied by the model implementation,
for example `model.layers.0.self_attn.attn` for the Llama implementation. VLM
prefixes may include `language_model`; inspect the model's attention construction
rather than guessing names. A missing name produces an error, never an empty
successful heatmap. `head_indices` select global query heads after TP gathering.

## Python interface

```python
from vllm import LLM, SamplingParams
from vllm.attention_diagnostics import AttentionDiagnosticsParams

# Launch with VLLM_USE_V2_MODEL_RUNNER=1 in the environment.
llm = LLM(model="your-model", enforce_eager=True, async_scheduling=False)

# Persist these IDs from the original inference, including hidden/special IDs.
replay_ids = original_prompt_token_ids + original_output_token_ids
params = SamplingParams(
    attention_diagnostics=AttentionDiagnosticsParams(
        query_positions=[3, len(replay_ids) - 1],
        layer_names=["model.layers.0.self_attn.attn"],
        head_indices=[0, 1],
        # Optional columns to return; normalization still uses ALL legal keys.
        key_positions=[0, 1, 2],
    ),
)
result = llm.generate({"prompt_token_ids": replay_ids}, params)[0]
assert result.outputs[0].token_ids == []
diagnostic = result.outputs[0].attention_diagnostics
if "error" in diagnostic:
    raise RuntimeError(diagnostic["error"])
weights = diagnostic["layers"]["model.layers.0.self_attn.attn"]["weights"]
# Shape: [selected_heads, selected_queries, selected_keys].
```

For multimodal replay, also preserve and supply the original media/processor
inputs using the existing multimodal input interface. A token ID sequence alone
cannot reconstruct image/video embeddings. Use the same model revision, adapter,
processor options, frame ordering, resolution, and timestamps. Verify processed
placeholder ranges before mapping selected columns back to image patches.
This extension returns token-level attention; image-grid restoration belongs to
the caller.

## HTTP interface

The existing `/inference/v1/generate` token-in/token-out endpoint accepts the
extension inside `sampling_params`. It is not an OpenAI chat/completions field.
Enable this endpoint when starting the coupled server:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve your-model \
  --enable-scale-out --enforce-eager --no-async-scheduling
```

Use `stream=false` and `output_mode="tokens"`:

```json
{
  "token_ids": [1, 42, 43, 44],
  "stream": false,
  "output_mode": "tokens",
  "sampling_params": {
    "attention_diagnostics": {
      "query_positions": [0, 2, 3],
      "layer_names": ["model.layers.0.self_attn.attn"],
      "key_positions": [0, 1, 3],
      "head_indices": [0]
    }
  }
}
```

Read `choices[0].attention_diagnostics`. Successful payloads contain
`sequence_length`, `query_positions`, `key_positions`, and `layers`, each with
`head_indices` and `weights`. `choices[0].token_ids` is empty and completion token
usage is zero. Capture failures return `{"error": "..."}` in that field; callers
must check it before rendering. Invalid request/configuration combinations are
rejected before scheduling. Streaming HTTP diagnostics and resumable streaming inputs are rejected.

For VLMs, preserve the existing `features` payload from the render/generate API,
including actual multimodal kwargs, hashes and placeholder metadata. This
extension does not add a media transport protocol. Retaining only hashes is not
sufficient if their backing cache entries expire.

## Bounds and correctness

- At most 128 query positions and 8 layers per request.
- CPU Q/K buffer default: 256 MiB per TP rank, hard limit: 1 GiB per rank.
- Returned/gathered tensor values: at most 1,000,000 across selected layers,
  counting all heads before optional head selection. JSON uses additional memory.
- Scores are computed in FP32, in key blocks, with online log-sum-exp over all
  legal keys. Selecting image-only columns does not renormalize image mass to 1.
- Future-key probabilities are zero. Summing a subset of columns can be below 1.
- Buffers are dropped on completion, abort or preemption. Preemption restarts
  capture from scratch because prefix reads are disabled.
- Use a replay sequence shorter than the server's `max_model_len`; the current
  generation input validator still reserves a slot even though diagnostics do
  not generate. Truncated input cannot reproduce a longer original context.

The reference softmax/collector tests can run without compiled vLLM GPU kernels:

```bash
uv run --no-project .venv/bin/python -m pytest \
  --confcutdir=tests/attention_diagnostics tests/attention_diagnostics
```

Run the scheduler regression with the complete vLLM development environment:

```bash
uv run --no-project .venv/bin/python -m pytest tests/v1/core/test_scheduler.py \
  -k attention_diagnostics
```

Before production use, compare small eager model runs against a reference
attention implementation, then validate TP and multimodal token/patch alignment.
A passing numerical unit test alone does not establish model-serving parity.
