# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, prefill-only attention diagnostics over logical input positions."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AttentionDiagnosticsParams:
    """Experimental prefill diagnostics, indexed in the processed token sequence."""

    query_positions: list[int]
    layer_names: list[str]
    key_positions: list[int] | None = None
    head_indices: list[int] | None = None
    max_buffer_bytes: int = 256 * 1024 * 1024
    max_output_values: int = 1_000_000
    early_exit: bool = True

    def validate(self, prompt_length: int | None = None) -> None:
        if type(self.early_exit) is not bool:
            raise ValueError("early_exit must be a boolean")
        for name in ("query_positions", "key_positions", "head_indices"):
            values = getattr(self, name)
            if values is None and name != "query_positions":
                continue
            if not values or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"{name} must contain non-negative integer positions")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            if (
                prompt_length is not None
                and name != "head_indices"
                and max(values) >= prompt_length
            ):
                raise ValueError(f"{name} exceeds the processed prompt length")
        if (
            not self.layer_names
            or len(set(self.layer_names)) != len(self.layer_names)
            or any(not isinstance(n, str) or not n for n in self.layer_names)
        ):
            raise ValueError("layer_names must contain unique, non-empty module names")
        if len(self.query_positions) > 128 or len(self.layer_names) > 8:
            raise ValueError("At most 128 query positions and 8 layers are supported")
        if (
            type(self.max_buffer_bytes) is not int
            or not 0 < self.max_buffer_bytes <= 1024**3
        ):
            raise ValueError("max_buffer_bytes must be in (0, 1 GiB]")
        if (
            type(self.max_output_values) is not int
            or not 0 < self.max_output_values <= 1_000_000
        ):
            raise ValueError("max_output_values must be in (0, 1000000]")


def selected_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    query_positions: list[int],
    key_positions: list[int],
    scale: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """Compute selected causal rows, normalized over ALL preceding keys.

    Args:
        q: Selected queries, [query_heads, selected_queries, head_dim].
        k: Complete keys, [kv_heads, sequence_length, head_dim].
        query_positions: Logical positions of the query rows.
        key_positions: Columns to return, without changing normalization.
        scale: The model layer's attention scale.
        block_size: Key block size for bounded score memory.

    Returns:
        FP32 probabilities [query_heads, selected_queries, selected_keys].

    """
    if block_size <= 0 or q.shape[0] % k.shape[0]:
        raise ValueError("Invalid block size or GQA head mapping")
    positions = torch.tensor(query_positions, device=q.device)
    heads = torch.arange(q.shape[0], device=q.device) // (q.shape[0] // k.shape[0])
    query = q.float()
    log_z = torch.full(q.shape[:2], -torch.inf, device=q.device)
    for start in range(0, k.shape[1], block_size):
        end = min(start + block_size, k.shape[1])
        keys = k[:, start:end].to(q.device).float()[heads]
        scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
        masked = torch.arange(start, end, device=q.device)[None, :] > positions[:, None]
        scores.masked_fill_(masked[None, :, :], -torch.inf)
        log_z = torch.logaddexp(log_z, torch.logsumexp(scores, dim=-1))
    result = torch.empty((*q.shape[:2], len(key_positions)), device=q.device)
    for start in range(0, len(key_positions), block_size):
        indices = key_positions[start : start + block_size]
        keys = k[:, indices].to(q.device).float()[heads]
        scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
        masked = torch.tensor(indices, device=q.device)[None, :] > positions[:, None]
        scores.masked_fill_(masked[None, :, :], -torch.inf)
        result[:, :, start : start + len(indices)] = torch.exp(
            scores - log_z[..., None]
        )
    return result


@dataclass
class _LayerCapture:
    keys: torch.Tensor
    queries: torch.Tensor
    keys_seen: torch.Tensor
    queries_seen: torch.Tensor
    scale: float
    device: torch.device


@dataclass
class _SparseCapture:
    weights: torch.Tensor
    queries_seen: torch.Tensor
    selected_positions: list[list[int]]
    device: torch.device


def sparse_attention_row(query, key_cache, packed, block_table, columns):
    """Read the actual QSA selection and paged keys, before the output gate."""
    count = int(packed[-1])
    if not 0 < count < len(packed):
        raise ValueError("Invalid QSA selection count")
    indices = packed[:count].long()
    page_size = key_cache.shape[1]
    if (indices < 0).any() or (indices // page_size >= len(block_table)).any():
        raise ValueError("Invalid QSA logical position")
    pages = block_table[indices // page_size].long()
    if (pages < 0).any() or (pages >= key_cache.shape[0]).any():
        raise ValueError("Invalid QSA physical page")
    keys = key_cache[pages, indices % page_size].float()
    heads = torch.arange(query.shape[0], device=query.device)
    heads = heads // (query.shape[0] // keys.shape[1])
    scores = torch.einsum("hd,khd->hk", query.float(), keys[:, heads])
    probabilities = (scores * query.shape[-1] ** -0.5).softmax(-1)
    # Aggregate duplicate positions if a backend ever emits them.
    result = query.new_empty((query.shape[0], len(columns)), dtype=torch.float32)
    for start in range(0, len(columns), 256):
        block = torch.tensor(columns[start : start + 256], device=query.device)
        matches = indices[:, None] == block[None, :]
        result[:, start : start + len(block)] = probabilities @ matches.float()
    return result, indices.cpu().tolist()


def capture_qsa(layer, query, key_cache, packed, block_table):
    active = _active_capture.get()
    if active is not None:
        collector, batch = active
        collector.capture_qsa(layer, query, key_cache, packed, block_table, batch)


@dataclass
class _RequestCapture:
    params: AttentionDiagnosticsParams
    length: int
    layers: dict[str, _LayerCapture | _SparseCapture] = field(default_factory=dict)
    error: str | None = None
    buffer_bytes: int = 0
    early_exit_layer: int | None = None


_active_capture: ContextVar[Any] = ContextVar("attention_diagnostics", default=None)


def capture_attention(layer: Any, query: torch.Tensor, key: torch.Tensor) -> None:
    active = _active_capture.get()
    if active is not None:
        collector, batch = active
        collector.capture(layer, query, key, batch)


def diagnostics_should_stop(layers: Any, layer_index: int) -> bool:
    """Stop supported decoder loops only after every requested layer ran.

    Resolve actual Attention prefixes, never parse user-supplied layer numbers.
    The scheduler guarantees exclusive replay; mixed batches fail closed here.
    All TP ranks make the same decision, independent of local capture errors.
    """
    active = _active_capture.get()
    if active is None:
        return False
    collector, batch = active
    states = [collector.requests.get(r) for r in batch.req_ids]
    if not states or any(s is None or not s.params.early_exit for s in states):
        return False
    names = {}
    for index, block in enumerate(layers):
        owner = getattr(block, "self_attn", None)
        attention = getattr(owner, "attn", owner)
        if attention is not None and hasattr(attention, "layer_name"):
            names[attention.layer_name] = index
    requested = {name for s in states for name in s.params.layer_names}
    if not requested.issubset(names):
        return False
    deepest = max(names[name] for name in requested)
    if layer_index != deepest or deepest == len(layers) - 1:
        return False
    for state in states:
        state.early_exit_layer = deepest
    return True


class AttentionDiagnosticsCollector:
    """Worker-local CPU buffers; no second model or persistent GPU Q/K copy."""

    def __init__(self) -> None:
        self.requests: dict[str, _RequestCapture] = {}

    def begin_step(self, scheduler_output: Any) -> None:
        for req_id in scheduler_output.finished_req_ids | (
            scheduler_output.preempted_req_ids or set()
        ):
            self.requests.pop(req_id, None)
        for req_id, (params, length) in scheduler_output.attention_diagnostics.items():
            if req_id not in self.requests:
                self.requests[req_id] = _RequestCapture(params, length)

    @contextmanager
    def context(self, batch: Any):
        if not self.requests:
            yield
            return
        token = _active_capture.set((self, batch))
        try:
            yield
        finally:
            _active_capture.reset(token)

    def capture(self, layer: Any, q: torch.Tensor, k: torch.Tensor, batch: Any) -> None:
        for i, req_id in enumerate(batch.req_ids):
            state = self.requests.get(req_id)
            if (
                state is None
                or state.error
                or layer.layer_name not in state.params.layer_names
            ):
                continue
            try:
                self._capture_layer(state, layer, q, k, batch, i)
            except (ValueError, RuntimeError, MemoryError) as exc:
                state.error = str(exc)
                state.layers.clear()

    def capture_qsa(self, layer, q, cache, packed, table, batch):
        for i, req_id in enumerate(batch.req_ids):
            state = self.requests.get(req_id)
            if (
                state is None
                or state.error
                or layer.layer_name not in state.params.layer_names
            ):
                continue
            try:
                params = state.params
                columns = (
                    params.key_positions
                    if params.key_positions is not None
                    else list(range(state.length))
                )
                entry = state.layers.get(layer.layer_name)
                if entry is None:
                    shape = (q.shape[1], len(params.query_positions), len(columns))
                    values = shape[0] * shape[1] * shape[2]
                    size = (
                        values * 4 + len(params.query_positions) * packed.shape[1] * 8
                    )
                    if (
                        values > params.max_output_values
                        or state.buffer_bytes + size > params.max_buffer_bytes
                    ):
                        raise ValueError("QSA capture exceeds diagnostic budget")
                    entry = _SparseCapture(
                        torch.zeros(shape),
                        torch.zeros(shape[1], dtype=torch.bool),
                        [[] for _ in params.query_positions],
                        q.device,
                    )
                    state.layers[layer.layer_name] = entry
                    state.buffer_bytes += size
                if not isinstance(entry, _SparseCapture):
                    raise ValueError("Attention capture type changed during replay")
                start, end = (int(x) for x in batch.query_start_loc_np[i : i + 2])
                offset = int(batch.num_computed_tokens_np[i])
                for j, position in enumerate(params.query_positions):
                    if offset <= position < offset + end - start:
                        row = start + position - offset
                        weights, selected = sparse_attention_row(
                            q[row], cache, packed[row], table[i], columns
                        )
                        if any(k > position for k in selected):
                            raise ValueError("QSA selection violates causality")
                        entry.weights[:, j] = weights.detach().cpu()
                        entry.selected_positions[j] = selected
                        entry.queries_seen[j] = True
            except (ValueError, RuntimeError, MemoryError) as exc:
                state.error = str(exc)
                state.layers.clear()

    def _capture_layer(self, state, layer, q, k, batch, i):
        if not layer.diagnostics_supported or k is None:
            raise ValueError(f"Unsupported attention semantics: {layer.layer_name}")
        start, end = (int(x) for x in batch.query_start_loc_np[i : i + 2])
        offset = int(batch.num_computed_tokens_np[i])
        if offset + end - start > state.length:
            raise ValueError("Attention diagnostics must not execute decode tokens")
        q = q[start:end].reshape(-1, layer.num_heads, layer.head_size)
        k = k[start:end].reshape(-1, layer.num_kv_heads, layer.head_size)
        params = state.params
        entry = state.layers.get(layer.layer_name)
        if entry is None:
            size = layer.head_size * (
                state.length * layer.num_kv_heads * k.element_size()
                + len(params.query_positions) * layer.num_heads * q.element_size()
            )
            size += state.length + len(params.query_positions)
            if state.buffer_bytes + size > params.max_buffer_bytes:
                raise ValueError("Attention capture exceeds max_buffer_bytes")
            entry = _LayerCapture(
                torch.empty(
                    (layer.num_kv_heads, state.length, layer.head_size), dtype=k.dtype
                ),
                torch.empty(
                    (layer.num_heads, len(params.query_positions), layer.head_size),
                    dtype=q.dtype,
                ),
                torch.zeros(state.length, dtype=torch.bool),
                torch.zeros(len(params.query_positions), dtype=torch.bool),
                layer.diagnostics_scale,
                q.device,
            )
            state.layers[layer.layer_name] = entry
            state.buffer_bytes += size
        entry.keys[:, offset : offset + len(k)] = k.transpose(0, 1).detach().cpu()
        entry.keys_seen[offset : offset + len(k)] = True
        for j, pos in enumerate(params.query_positions):
            if offset <= pos < offset + len(q):
                entry.queries[:, j] = q[pos - offset].detach().cpu()
                entry.queries_seen[j] = True

    def finish_step(self, batch: Any) -> dict[str, dict]:
        results = {}
        for i, req_id in enumerate(batch.req_ids):
            state = self.requests.get(req_id)
            if state is None:
                continue
            end = int(batch.num_computed_tokens_np[i] + batch.num_scheduled_tokens[i])
            if end < state.length:
                continue
            try:
                results[req_id] = self._finish(state)
            finally:
                del self.requests[req_id]
        return results

    def _finish(self, state: _RequestCapture) -> dict:
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        errors: list[str | None] = [None] * group.world_size
        error = state.error
        if not error:
            for name in state.params.layer_names:
                entry = state.layers.get(name)
                if (
                    entry is None
                    or (isinstance(entry, _LayerCapture) and not entry.keys_seen.all())
                    or not entry.queries_seen.all()
                ):
                    error = f"Incomplete attention capture for {name}"
                    break
        if group.world_size > 1:
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
        else:
            errors[0] = error
        if any(errors):
            return {"error": next(e for e in errors if e)}
        params = state.params
        keys = (
            params.key_positions
            if params.key_positions is not None
            else list(range(state.length))
        )
        result: dict[str, Any] = {
            "query_positions": params.query_positions,
            "key_positions": keys,
            "sequence_length": state.length,
            "early_exit_layer": state.early_exit_layer,
            "layers": {},
        }
        total_values = 0
        for name in params.layer_names:
            entry = state.layers[name]
            num_heads = (
                entry.weights.shape[0]
                if isinstance(entry, _SparseCapture)
                else entry.queries.shape[0]
            ) * group.world_size
            if (
                params.head_indices is not None
                and max(params.head_indices) >= num_heads
            ):
                return {"error": f"head_indices exceeds head count for {name}"}
            # Bound the gathered tensor too, even if the caller selects fewer heads.
            total_values += num_heads * len(params.query_positions) * len(keys)
            if total_values > params.max_output_values:
                return {"error": "Attention output exceeds max_output_values"}
            weights = None
            error = None
            try:
                weights = (
                    entry.weights.to(entry.device)
                    if isinstance(entry, _SparseCapture)
                    else selected_attention(
                        entry.queries.to(entry.device),
                        entry.keys,
                        params.query_positions,
                        keys,
                        entry.scale,
                    )
                )
                if not torch.isfinite(weights).all():
                    error = f"Non-finite attention values for {name}"
            except (ValueError, RuntimeError, MemoryError) as exc:
                error = str(exc)
            # Every TP rank must reach the same collectives, even if only one
            # rank ran out of scratch space while recomputing its head shard.
            if group.world_size > 1:
                torch.distributed.all_gather_object(
                    errors, error, group=group.cpu_group
                )
            else:
                errors[0] = error
            if any(errors):
                return {"error": next(e for e in errors if e)}
            assert weights is not None
            if group.world_size > 1:
                weights = group.all_gather(weights, dim=0)
            heads = (
                params.head_indices
                if params.head_indices is not None
                else list(range(num_heads))
            )
            result["layers"][name] = {
                "head_indices": heads,
                "weights": weights[heads].cpu().tolist(),
            }
            if isinstance(entry, _SparseCapture):
                result["layers"][name]["attention_type"] = "qsa"
                result["layers"][name]["selected_key_positions"] = (
                    entry.selected_positions
                )
                result["layers"][name]["weight_stage"] = "before_output_gate"
        return result
