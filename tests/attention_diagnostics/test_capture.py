# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and lifetime tests that need neither model weights nor CUDA kernels."""

from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from vllm.attention_diagnostics import (
    AttentionDiagnosticsCollector,
    AttentionDiagnosticsParams,
    capture_attention,
    selected_attention,
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("block_size", [1, 3, 64])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_selected_rows_match_full_causal_gqa(block_size, kv_heads, device):
    """Selecting columns must not renormalize away the unselected keys."""
    torch.manual_seed(42)
    q = torch.randn(4, 9, 8, device=device)
    k = torch.randn(kv_heads, 9, 8, device=device)
    positions, columns = [8, 0, 4], [7, 0, 3]
    scores = q @ k.repeat_interleave(4 // kv_heads, 0).transpose(-1, -2) / 8**0.5
    scores.masked_fill_(
        torch.ones(9, 9, dtype=torch.bool, device=device).triu(1), -torch.inf
    )
    expected = scores.softmax(-1)[:, positions][:, :, columns]
    actual = selected_attention(
        q[:, positions], k, positions, columns, 8**-0.5, block_size
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[:, 1, [0, 2]], torch.zeros(4, 2, device=device))
    assert torch.all(actual[:, 0].sum(-1) < 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_positions": []},
        {"query_positions": [-1]},
        {"query_positions": [True]},
        {"query_positions": [1, 1]},
        {"query_positions": [8]},
        {"key_positions": [8]},
        {"head_indices": []},
        {"layer_names": []},
        {"max_buffer_bytes": 0},
        {"max_buffer_bytes": 2**30 + 1},
        {"max_output_values": True},
    ],
)
def test_invalid_selection_is_rejected(kwargs):
    args = {"query_positions": [0], "layer_names": ["layer"]} | kwargs
    with pytest.raises(ValueError):
        AttentionDiagnosticsParams(**args).validate(8)


def test_diagnostic_params_survive_http_and_engine_serialization():
    raw = {"query_positions": [3, 1], "key_positions": [0, 2], "layer_names": ["layer"]}
    params = TypeAdapter(AttentionDiagnosticsParams).validate_python(raw)
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(params), type=AttentionDiagnosticsParams
    )
    assert restored == params
    restored.validate(4)


def batch(offset, length):
    return SimpleNamespace(
        req_ids=["r"],
        query_start_loc_np=np.array([0, length]),
        num_computed_tokens_np=np.array([offset]),
        num_scheduled_tokens=np.array([length]),
    )


def start(collector, params, length=6, **kwargs):
    collector.begin_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            attention_diagnostics={"r": (params, length)},
            **kwargs,
        )
    )


def layer():
    return SimpleNamespace(
        layer_name="layer",
        num_heads=4,
        num_kv_heads=2,
        head_size=8,
        diagnostics_supported=True,
        diagnostics_scale=8**-0.5,
    )


def test_chunked_capture_retains_arbitrary_queries_and_resets_context():
    torch.manual_seed(1)
    q, k = torch.randn(6, 32), torch.randn(6, 16)
    params = AttentionDiagnosticsParams([5, 1], ["layer"], head_indices=[3, 0])
    collector = AttentionDiagnosticsCollector()
    start(collector, params)
    with collector.context(batch(0, 3)):
        capture_attention(layer(), q[:3], k[:3])
    assert collector.finish_step(batch(0, 3)) == {}
    # Outside the scope, unrelated inference must not overwrite a capture.
    capture_attention(layer(), q[:3] * 0, k[:3] * 0)
    with collector.context(batch(3, 3)):
        capture_attention(layer(), q[3:], k[3:])
    # Patch only the distributed group, leaving the numerical path real.
    with patch.dict(
        "sys.modules",
        {
            "vllm.distributed": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(world_size=1)
            )
        },
    ):
        result = collector.finish_step(batch(3, 3))["r"]
    expected = selected_attention(
        q.reshape(6, 4, 8).transpose(0, 1)[:, [5, 1]],
        k.reshape(6, 2, 8).transpose(0, 1),
        [5, 1],
        list(range(6)),
        8**-0.5,
    )
    torch.testing.assert_close(
        torch.tensor(result["layers"]["layer"]["weights"]), expected[[3, 0]]
    )
    assert not collector.requests


@pytest.mark.parametrize("unsupported", [False, True])
def test_failed_capture_returns_error_and_releases_buffers(unsupported):
    params = AttentionDiagnosticsParams([1], ["layer"], max_buffer_bytes=1)
    collector = AttentionDiagnosticsCollector()
    start(collector, params, 2)
    attn = layer()
    attn.diagnostics_supported = not unsupported
    with collector.context(batch(0, 2)):
        capture_attention(attn, torch.zeros(2, 32), torch.zeros(2, 16))
    with patch.dict(
        "sys.modules",
        {
            "vllm.distributed": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(world_size=1)
            )
        },
    ):
        result = collector.finish_step(batch(0, 2))["r"]
    assert "error" in result
    assert "weights" not in result
    assert not collector.requests


def test_preempted_and_aborted_requests_discard_partial_capture():
    collector = AttentionDiagnosticsCollector()
    params = AttentionDiagnosticsParams([1], ["layer"])
    start(collector, params)
    old = collector.requests["r"]
    collector.begin_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"r"},
            attention_diagnostics={"r": (params, 6)},
        )
    )
    assert collector.requests["r"] is not old
    collector.begin_step(
        SimpleNamespace(
            finished_req_ids={"r"}, preempted_req_ids=None, attention_diagnostics={}
        )
    )
    assert not collector.requests


@pytest.mark.parametrize(
    "failure", ["missing_layer", "output_budget", "head_count", "nan"]
)
def test_invalid_results_fail_closed(failure):
    params = AttentionDiagnosticsParams([1], ["layer"])
    if failure == "output_budget":
        params.max_output_values = 1
    elif failure == "head_count":
        params.head_indices = [4]
    collector = AttentionDiagnosticsCollector()
    start(collector, params, 2)
    q = torch.zeros(2, 32)
    if failure == "nan":
        q.fill_(torch.nan)
    if failure != "missing_layer":
        with collector.context(batch(0, 2)):
            capture_attention(layer(), q, torch.zeros(2, 16))
    with patch.dict(
        "sys.modules",
        {
            "vllm.distributed": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(world_size=1)
            )
        },
    ):
        result = collector.finish_step(batch(0, 2))["r"]
    assert "error" in result
    assert "layers" not in result
    assert not collector.requests


@pytest.mark.parametrize(
    "enabled, names, stop",
    [
        (True, ["decoder.0", "decoder.2"], 2),
        (True, ["decoder.0"], 0),
        (False, ["decoder.0"], None),
        (True, ["missing"], None),
        (True, ["decoder.3"], None),
    ],
)
def test_early_exit_uses_deepest_resolved_layer(enabled, names, stop):
    from vllm.attention_diagnostics import diagnostics_should_stop

    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(attn=SimpleNamespace(layer_name=f"decoder.{i}"))
        )
        for i in range(4)
    ]
    collector = AttentionDiagnosticsCollector()
    collector.begin_step(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            attention_diagnostics={
                "r": (
                    AttentionDiagnosticsParams(
                        query_positions=[0], layer_names=names, early_exit=enabled
                    ),
                    1,
                )
            },
        )
    )
    batch = SimpleNamespace(req_ids=["r"])
    with collector.context(batch):
        assert [i for i in range(4) if diagnostics_should_stop(layers, i)] == (
            [] if stop is None else [stop]
        )
    assert not diagnostics_should_stop(layers, 0)
    with collector.context(SimpleNamespace(req_ids=["r", "ordinary"])):
        assert not diagnostics_should_stop(layers, 0)


def test_diagnostics_never_publish_prefix_blocks():
    from unittest.mock import Mock

    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    coordinator = Mock()
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            attention_diagnostics=AttentionDiagnosticsParams(
                query_positions=[0], layer_names=["layer"]
            )
        )
    )
    KVCacheCoordinator.cache_blocks(coordinator, request, 32)
    coordinator.get_replay_boundaries.assert_not_called()


def test_qsa_paged_selection_preserves_sparse_normalization():
    """Count is not an index; unselected positions stay zero across page remaps."""
    from vllm.attention_diagnostics import sparse_attention_row

    torch.manual_seed(5)
    q = torch.randn(4, 8)
    cache = torch.randn(3, 2, 2, 8)
    table = torch.tensor([2, 0, 1])
    packed = torch.tensor([0, 3, 4, -1, 3])
    actual, selected = sparse_attention_row(q, cache, packed, table, [4, 1, 0])
    keys = torch.stack([cache[2, 0], cache[0, 1], cache[1, 0]])
    scores = torch.einsum("hd,khd->hk", q, keys.repeat_interleave(2, 1)) / 8**0.5
    expected = scores.softmax(-1)
    torch.testing.assert_close(actual[:, 0], expected[:, 2])
    torch.testing.assert_close(actual[:, 2], expected[:, 0])
    assert actual[:, 1].count_nonzero() == 0
    assert selected == [0, 3, 4]


def test_qsa_chunked_capture_and_early_exit_owner():
    from vllm.attention_diagnostics import capture_qsa, diagnostics_should_stop

    collector = AttentionDiagnosticsCollector()
    start(collector, AttentionDiagnosticsParams([5, 1], ["qsa"]))
    owner = SimpleNamespace(layer_name="qsa")
    layers = [SimpleNamespace(self_attn=owner), SimpleNamespace()]
    cache = torch.ones(3, 2, 2, 8)
    table = torch.tensor([[2, 0, 1]])
    for offset in [0, 3]:
        packed = torch.tensor([[0, -1, 1], [0, -1, 1], [0, offset + 2, 2]])
        with collector.context(batch(offset, 3)):
            capture_qsa(owner, torch.ones(3, 4, 8), cache, packed, table)
            assert diagnostics_should_stop(layers, 0)
    with patch.dict(
        "sys.modules",
        {
            "vllm.distributed": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(world_size=1)
            )
        },
    ):
        result = collector.finish_step(batch(3, 3))["r"]
    weights = torch.tensor(result["layers"]["qsa"]["weights"])
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, 2))
    assert result["layers"]["qsa"]["selected_key_positions"] == [[0, 5], [0]]
    assert result["early_exit_layer"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_probabilities_reconstruct_kernel_output():
    """Compare diagnostic weights against the actual paged QSA GPU kernel."""
    from vllm.attention_diagnostics import sparse_attention_row
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    torch.manual_seed(8)
    q = torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    packed = torch.full((1, 33), -1, device="cuda", dtype=torch.int32)
    packed[0, :3] = torch.tensor([0, 19, 37], device="cuda")
    packed[0, -1] = 3
    table = torch.tensor([[2, 0, 1]], device="cuda", dtype=torch.int32)
    gate = torch.zeros_like(q)
    output = qsa_sparse_paged_attention(
        q,
        k,
        v,
        packed,
        table,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        True,
        output_gate=gate,
    )
    weights, _ = sparse_attention_row(q[0], k, packed[0], table[0], [0, 19, 37])
    values = torch.stack([v[2, 0], v[0, 3], v[1, 5]]).repeat_interleave(2, 1)
    expected = torch.einsum("hk,khd->hd", weights, values.float()) * 0.5
    torch.testing.assert_close(output[0].float(), expected, atol=0.015, rtol=0.015)
