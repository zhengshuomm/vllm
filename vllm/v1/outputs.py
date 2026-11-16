# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.connector import AuxRequestOutput
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorWorkerMetadata
    from vllm.distributed.ec_transfer.ec_connector.metrics import ECConnectorStats
    from vllm.distributed.kv_events import KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
else:
    AuxRequestOutput = object
    KVConnectorStats = object
    KVConnectorWorkerMetadata = object
    KVConnectorKVEvents = object
    ECConnectorWorkerMetadata = object


class LogprobsLists(NamedTuple):
    """序列化回传给 EngineCore（CPU 进程）的扁平化对数概率列表。

    由于投机解码（Speculative Decoding）中不同请求单步生成的 token 数可能不同，
    这里将所有请求生成的各个位置的 top-k logprobs 沿第 0 维拼成紧凑二维数组，
    并用前缀和 `cu_num_generated_tokens` 记录每个请求对应的行区间。

    Sample（假设 batch 有 3 个请求：`req_B` 投机生成 2 个 token，`req_A` 生成 1 个 token，
    `req_C` 仍在 Chunked Prefill 生成 0 个 token；用户设置 `logprobs=2`，故每行保留
    `max_num_logprobs + 1 = 3` 列，即「采样出的 token 自身 + top-2 候选 token」）：
        LogprobsLists(
            # shape = (3, 3) -> 共 2 + 1 + 0 = 3 个生成位置
            logprob_token_ids=np.array([
                [502, 502, 610],   # req_B 第 0 个生成位置：采样出 502，top-2 为 502, 610
                [999, 503, 999],   # req_B 第 1 个生成位置：采样出 999，top-2 为 503, 999
                [777, 777, 128],   # req_A 第 0 个生成位置：采样出 777，top-2 为 777, 128
            ], dtype=np.int32),
            logprobs=np.array([
                [-0.05, -0.05, -3.20],
                [-0.80, -0.65, -0.80],
                [-0.01, -0.01, -4.90],
            ], dtype=np.float32),
            # 每个生成位置实际采样出的 token 在词表概率降序中的名次（0-based）
            sampled_token_ranks=np.array([0, 1, 0], dtype=np.int32),
            # 请求级行偏移前缀和：[0, 2, 3, 3]
            # -> req_B 取 [0:2]，req_A 取 [2:3]，req_C 取 [3:3]（空）
            cu_num_generated_tokens=[0, 2, 3, 3],
        )
    """

    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


class SamplingMaskLists(NamedTuple):
    """CSR sampling masks; a request slice may contain multiple positions."""

    # [num_kept_tokens]
    token_ids: np.ndarray
    # [num_positions + 1], or None for a single position
    offsets: np.ndarray | None = None
    # [num_requests + 1] for multi-position request batches.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int) -> "SamplingMaskLists":
        assert self.offsets is not None
        cu = self.cu_num_generated_tokens
        start = req_idx if cu is None else cu[req_idx]
        end = start + num_positions
        lo, hi = self.offsets[start], self.offsets[end]
        if num_positions == 1:
            return SamplingMaskLists(self.token_ids[lo:hi])
        return SamplingMaskLists(
            self.token_ids[lo:hi], self.offsets[start : end + 1] - lo
        )

    def to_nested_list(self) -> list[list[int]]:
        token_ids = self.token_ids.tolist()
        if self.offsets is None:
            return [token_ids]
        offsets = self.offsets.tolist()
        return [token_ids[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


class LogprobsTensors(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # [num_reqs + 1]
    cu_num_generated_tokens: list[int] | None = None
    # [num_reqs + 1]. Set instead of cu_num_generated_tokens when the
    # boundaries only exist on device (adaptive verification); rides along on
    # the async D2H copy.
    cu_num_generated_tokens_tensor: torch.Tensor | None = None

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        if cu_num_generated_tokens is None:
            if self.cu_num_generated_tokens_tensor is not None:
                cu_num_generated_tokens = self.cu_num_generated_tokens_tensor.tolist()
            else:
                cu_num_generated_tokens = self.cu_num_generated_tokens
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        cu_tensor = self.cu_num_generated_tokens_tensor
        if cu_tensor is not None:
            cu_tensor = cu_tensor.to("cpu", non_blocking=True)
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
            cu_tensor,
        )

    def filter(self, mask: torch.Tensor) -> "LogprobsTensors":
        """Filter the logprobs tensors with the given bool mask."""
        assert self.cu_num_generated_tokens is None, (
            "filter can't be used with cu_num_generated_tokens"
        )
        assert self.cu_num_generated_tokens_tensor is None, (
            "filter can't be used with cu_num_generated_tokens_tensor"
        )
        return LogprobsTensors(
            self.logprob_token_ids[mask],
            self.logprobs[mask],
            self.selected_token_ranks[mask],
        )

    @staticmethod
    def cat(
        tensors: Sequence["LogprobsTensors"],
        cu_num_generated_tokens: list[int] | None = None,
    ) -> "LogprobsTensors":
        """Concatenate flattened logprob tensors."""
        assert tensors
        assert cu_num_generated_tokens is not None or all(
            tensor.cu_num_generated_tokens is None for tensor in tensors
        )
        if len(tensors) == 1:
            tensor = tensors[0]
            if cu_num_generated_tokens is None:
                return tensor
            return tensor._replace(cu_num_generated_tokens=cu_num_generated_tokens)
        # The multi-chunk path rebuilds boundaries from the CPU layout, which
        # the device-only boundaries of adaptive verification never use.
        assert all(tensor.cu_num_generated_tokens_tensor is None for tensor in tensors)
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [tensor.logprob_token_ids for tensor in tensors]
            ),
            logprobs=torch.cat([tensor.logprobs for tensor in tensors]),
            selected_token_ranks=torch.cat(
                [tensor.selected_token_ranks for tensor in tensors]
            ),
            cu_num_generated_tokens=cu_num_generated_tokens,
        )

    @staticmethod
    def empty_cpu(
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""
        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        logprobs = logprob_token_ids.new_empty(
            (num_positions, num_tokens_per_position),
            dtype=torch.float32,
            pin_memory=PIN_MEMORY,
        )
        selected_token_ranks = logprob_token_ids.new_empty(
            num_positions, pin_memory=PIN_MEMORY
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


# [num_reqs, <dynamic>]
# The shape of each element depends on the pooler used
PoolerOutput: TypeAlias = torch.Tensor | list[torch.Tensor] | list[torch.Tensor | None]


@dataclass
class SamplerOutput:
    """GPU 采样器（Sampler / RejectionSampler）的原始设备端输出张量。

    Sample（接上例 3 个请求，最大单步生成数 `max_num_generated_tokens = 3`，即 2 个草稿 + 1 个 bonus）：
        SamplerOutput(
            # shape = [3, 3]（驻留在 GPU 上，不足处用 PLACEHOLDER_TOKEN_ID = -1 填充）：
            # - row 0 (req_B): 接受草稿 502，拒绝第 2 个草稿并重采样出 999 -> [502, 999, -1]
            # - row 1 (req_A): 常规 Decode 采样出 777                      -> [777,  -1, -1]
            # - row 2 (req_C): Chunked Prefill 尚未扫完 prompt，虽算出末位 token 404，
            #                  但配合 num_sampled=[2, 1, 0] 在 D2H 解包时会被截断为空 []
            sampled_token_ids=tensor([
                [502, 999,  -1],
                [777,  -1,  -1],
                [404,  -1,  -1],
            ], device='cuda:0', dtype=torch.int32),
            logprobs_tensors=LogprobsTensors(...)  # 若全部请求均未开启 logprobs 则为 None
        )
    """

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


@dataclass
class KVConnectorOutput:
    # [req_ids]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    kv_connector_stats: KVConnectorStats | None = None
    kv_cache_events: KVConnectorKVEvents | None = None
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    invalid_block_ids: set[int] = field(default_factory=set)
    # Receive failures keyed by request identity. This remains unambiguous for
    # hybrid/multi-pool cache layouts where numeric block IDs overlap.
    failed_recving: set[str] = field(default_factory=set)
    # Configuration describing how many finished sending/receiving
    # notifications should be expected for each request. This allows
    # handshake-based connectors like Nixl to update the KVOutputAggregator.
    # It captures a static setup info and should almost always remain constant
    # for a given connector after discovery. Default value entails no change.
    expected_finished_count: int = 0

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.invalid_block_ids
            and not self.failed_recving
            and not self.kv_connector_worker_meta
        )


@dataclass
class ECConnectorOutput:
    # [mm_hash]
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    ec_connector_stats: "ECConnectorStats | None" = None
    ec_connector_worker_meta: ECConnectorWorkerMetadata | None = None

    def is_empty(self):
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.ec_connector_stats
            and not self.ec_connector_worker_meta
        )


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass
class ModelRunnerOutput:
    """Worker（仅 output_rank 进程）经共享内存 / ZMQ IPC 返回给 EngineCore Scheduler 的单步执行结果。

    注意：为了避免跨进程序列化 `torch.Tensor` 触发高昂的 CUDA IPC 句柄交换开销，
    `sampled_token_ids` 已在 Worker 的 `AsyncOutput.get_output()` 中通过异步 D2H 拷贝
    并按 `num_sampled` 截断为纯 Python 嵌套列表 `list[list[int]]`。

    Sample（3 请求混合批次 `[req_B(投机), req_A(常规Decode), req_C(Chunked Prefill未完)]`）：
        ModelRunnerOutput(
            req_ids=["req_B", "req_A", "req_C"],
            req_id_to_index={"req_B": 0, "req_A": 1, "req_C": 2},
            # 变长嵌套列表：每个子列表仅包含本步真正采纳的有效新 token
            sampled_token_ids=[
                [502, 999],  # req_B：投机解码本步产出 2 个新 token
                [777],       # req_A：常规 Decode 本步产出 1 个新 token
                [],          # req_C：Chunked Prefill 尚未读完整个 prompt，本步产出 0 个 token
            ],
            logprobs=LogprobsLists(...),       # 未请求 logprobs 时为 None
            prompt_logprobs_dict={},           # 仅当请求开启 prompt_logprobs 时填充对应 req_id
            pooler_output=None,                # 仅 Embedding / Rerank 等 Pooling 模型非空
            kv_connector_output=None,          # P/D 分离场景下回报异步 KV 发送/接收完成的 req_id 集合
        )
    """

    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # req_id -> [num_scored_rows, num_token_ids] prompt_logprob_token_ids scores.
    prompt_token_id_logprobs_dict: dict[str, torch.Tensor] = field(default_factory=dict)

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    aux_output_connector_output: dict[str, AuxRequestOutput] | None = None

    # ``None`` when ``return_sampling_mask`` is off.
    sampling_masks: SamplingMaskLists | None = None

    @staticmethod
    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def with_ec_conn_output_only(
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return an otherwise-empty output carrying `ec_connector_output`."""
        return ModelRunnerOutput.with_ec_conn_output(
            EMPTY_MODEL_RUNNER_OUTPUT, ec_connector_output
        )

    @staticmethod
    def with_ec_conn_output(
        output: "ModelRunnerOutput",
        ec_connector_output: ECConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return `output` carrying `ec_connector_output`.

        The shared empty output is copied rather than written to, so callers
        must use the return value.
        """
        if ec_connector_output is None or ec_connector_output.is_empty():
            return output
        if output is EMPTY_MODEL_RUNNER_OUTPUT:
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.ec_connector_output = ec_connector_output
        return output


# ModelRunnerOutput wrapper for async scheduling.
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass

    @abstractmethod
    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        """Get the underlying kv_connector_output.

        When get_output() fails, this is used to salvage the step's KV
        output instead of losing it with the failure.
        """
        pass


@dataclass
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]


def make_empty_encoder_model_runner_output(
    scheduler_output: "SchedulerOutput",
) -> ModelRunnerOutput:
    """Create a ModelRunnerOutput stub that contains the correct
    per-request bookkeeping but no generated data yet.
    """
    if not scheduler_output.num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # Convert to list so we get a deterministic, indexable sequence
    req_ids: list[str] = list(scheduler_output.num_scheduled_tokens.keys())

    # Give every request its own contiguous index
    req_id_to_index: dict[str, int] = {rid: idx for idx, rid in enumerate(req_ids)}

    # An encoder instance never samples, so it emits no tokens at all. The
    # scheduler finishes these requests once their prompt is fully encoded
    # (see `Scheduler.update_from_output`).
    sampled_token_ids: list[list[int]] = [[] for _ in req_ids]

    # Pooler outputs are not available yet ⇒ use None placeholders
    pooler_output: list[torch.Tensor | None] = [None for _ in req_ids]

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        pooler_output=pooler_output,
    )


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
