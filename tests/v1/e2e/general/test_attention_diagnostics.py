# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real runner/reference parity; uses a tiny local checkpoint, no model download."""

import pytest
import torch

from vllm import SamplingParams
from vllm.attention_diagnostics import AttentionDiagnosticsParams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_replay_matches_reference_and_does_not_generate(
    tmp_path, monkeypatch, vllm_runner
):
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    reference = LlamaForCausalLM(config).eval()
    reference.save_pretrained(tmp_path)
    ids = list(range(1, 22))
    positions, keys = [20, 0, 5], [0, 2, 6, 20]
    with torch.inference_mode():
        expected = reference(torch.tensor([ids]), output_attentions=True).attentions
    with vllm_runner(
        str(tmp_path),
        skip_tokenizer_init=True,
        enforce_eager=True,
        async_scheduling=False,
        dtype="float32",
        max_model_len=32,
        max_num_batched_tokens=4,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.15,
        attention_config={"backend": "TRITON_ATTN"},
    ) as runner:
        llm = runner.llm
        prompt = {"prompt_token_ids": ids}
        normal = SamplingParams(temperature=0, max_tokens=2, detokenize=False)
        # Replay first: a warm normal prefix would hide incomplete-cache pollution.
        early_params = SamplingParams(
            attention_diagnostics=AttentionDiagnosticsParams(
                query_positions=positions,
                key_positions=keys,
                layer_names=["model.layers.0.self_attn.attn"],
            )
        )
        early = llm.generate(prompt, early_params)[0].outputs[0]
        assert early.token_ids == []
        assert early.attention_diagnostics["early_exit_layer"] == 0
        torch.testing.assert_close(
            torch.tensor(
                early.attention_diagnostics["layers"]["model.layers.0.self_attn.attn"][
                    "weights"
                ]
            ),
            expected[0][0][:, positions][:, :, keys],
            atol=1e-4,
            rtol=1e-3,
        )
        before = llm.generate(prompt, normal)[0].outputs[0].token_ids
        with torch.inference_mode():
            reference_ids = list(ids)
            for _ in range(2):
                logits = reference(torch.tensor([reference_ids])).logits
                reference_ids.append(int(logits[0, -1].argmax()))
        assert list(before) == reference_ids[-2:]
        params = SamplingParams(
            attention_diagnostics=AttentionDiagnosticsParams(
                query_positions=positions,
                key_positions=keys,
                layer_names=[f"model.layers.{i}.self_attn.attn" for i in range(2)],
            )
        )
        output = llm.generate(prompt, params)[0].outputs[0]
        assert output.token_ids == []
        assert output.finish_reason == "stop"
        diagnostic = output.attention_diagnostics
        assert "error" not in diagnostic
        for i, ref in enumerate(expected):
            actual = torch.tensor(
                diagnostic["layers"][f"model.layers.{i}.self_attn.attn"]["weights"]
            )
            torch.testing.assert_close(
                actual, ref[0][:, positions][:, :, keys], atol=1e-4, rtol=1e-3
            )
        after = llm.generate(prompt, normal)[0].outputs[0].token_ids
        assert before == after
