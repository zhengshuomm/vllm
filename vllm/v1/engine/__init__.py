# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from vllm.config.kv_events import KVEventsConfig
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.outputs import RequestError
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_hints import KvHintsEnvelope
from vllm.v1.metrics.stats import (
    PrefillStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplingMaskLists
from vllm.v1.serial_utils import UtilityResult

# Type for pause_generation mode parameter.
# - "abort": Abort all in-flight requests immediately (default).
# - "wait": Wait for in-flight requests to complete before pausing.
# - "keep": Freeze requests in queue; they resume on resume_generation().
PauseMode = Literal["abort", "wait", "keep"]

# These are possible values of RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")

EEP_NOTIFICATION_CALL_ID = -1

FT_STATUS_CALL_ID = -2


class CheckpointState(enum.StrEnum):
    """Process checkpoint lifecycle of an engine. It serves only while
    RUNNING, and may go through the cycle any number of times."""

    RUNNING = "running"
    # Draining, then releasing what a checkpoint must not contain.
    PREPARING = "preparing"
    # Safe to capture.
    PREPARED = "prepared"
    # Bringing back what preparing released.
    RESTORING = "restoring"
    FAILED = "failed"

    @property
    def is_restored(self) -> bool:
        """Whether released state is back or coming back, so memory may wake and
        scheduling resume."""
        return self in (CheckpointState.RUNNING, CheckpointState.RESTORING)


class EEPNotificationType(enum.Enum):
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


class FinishReason(enum.IntEnum):
    """Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


@dataclass
class EngineCoreReadyResponse:
    """Sent from EngineCore to each frontend at the end of engine startup.

    Contains post-initialization config that may differ from the original
    values (e.g. max_model_len after KV cache auto-fitting).
    """

    max_model_len: int
    num_gpu_blocks: int
    block_size: int
    dp_stats_address: str | None
    dtype: str
    vllm_version: str
    world_size: int
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    decode_context_parallel_size: int
    data_parallel_rank: int
    max_num_seqs: int
    max_num_batched_tokens: int
    instance_id: str
    supports_lora: bool
    max_loras: int
    mamba_block_size: int | None = None
    # KV cache capacity (None for encoder-only/attention-free models).
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: KVEventsConfig | None = None
    weight_transfer_backend: str | None = None
    enable_sleep_mode: bool = False
    supports_draft_weight_updates: bool = False
    # Full-attention block size in tokens after initialization, or unavailable.
    effective_attention_block_size: int | None = None
    coord_store_port: int = 0
    coordinator_input_address: str | None = None
    coordinator_output_address: str | None = None
    # JSON-mode dump of this engine's VllmConfig, for frontends that serve
    # `/server_info` without holding the config themselves (the Rust frontend).
    # Sent only with VLLM_SERVER_DEV_MODE, the same gate as `/server_info`.
    vllm_config: dict[str, Any] | None = None


class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """前端 InputProcessor 构造并经 ZMQ 发送给后端 EngineCore 的标准化请求载荷（使用 msgspec 紧凑数组序列化）。

    多模态 + 文本混合请求的典型实例长这样（Sample）：
    ```python
    EngineCoreRequest(
        request_id="cmpl-8f3a-0",                  # 内部唯一 ID（若并行采样 n=2，会裂变为 ...-0, ...-1）
        external_req_id="cmpl-8f3a",               # 用户侧原始 request_id，用于外部 abort 映射
        prompt_token_ids=[128000, 3923, 374,       # 已展开多模态占位符后的完整 Token 序列
                          151655, ..., 151655,     # 连续 256 个 <image_pad> 占位 token (offset=3, length=256)
                          198, 264, 5629],
        mm_features=[
            MultiModalFeatureSpec(
                data={"pixel_values": Tensor([1, 3, 448, 448])},
                modality="image",
                identifier="9e107d9d...",          # 图像内容哈希（mm_hash），用于 EncoderCache 跨请求复用视觉特征
                mm_position=PlaceholderRange(offset=3, length=256),
            )
        ],
        sampling_params=SamplingParams(
            temperature=0.7, top_p=0.95, max_tokens=512,
            stop_token_ids={128001, 128009},       # 已由 InputProcessor 补全 tokenizer/generation_config 停止符
        ),
        pooling_params=None,                       # 生成任务为 None；Embedding/Rerank 任务则填 PoolingParams
        arrival_time=1730188900.42,
        lora_request=None,                         # 若挂载 LoRA 则为 LoRARequest(lora_name="sql-lora", lora_int_id=2, ...)
        cache_salt=None,                           # 多租户隔离盐值（混入 Prefix Cache 第一个块的哈希防侧信道探测）
        data_parallel_rank=None,
        client_index=0,                            # 多 API Server 实例扩展时，标记回包应路由给第几个前端进程
    )
    ```
    """

    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    abort_immediately: bool = False

    session_id: str | None = None
    kv_hints: KvHintsEnvelope | None = None

    @property
    def params(self) -> SamplingParams | PoolingParams:
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


class EngineCoreEventType(enum.IntEnum):
    """The type of engine core request event."""

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamp and is used by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: PrefillStats | None = None

    routed_experts: np.ndarray | None = None
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0
    # Multi-modal hashes missing from the P1 receiver cache (P0/P1 drift; see
    # `MultiModalCacheMissError`). Non-empty => retryable: the frontend drops these
    # from its sender cache and the request is resent with the data. Appended last
    # so `array_like` positional serialization stays backward compatible.
    mm_cache_miss_hashes: list[str] | None = None

    new_sampling_mask: SamplingMaskLists | None = None

    # Per-request spec-decode acceptance; attached only on the final output.
    # Appended last so `array_like` positional serialization stays compatible.
    spec_decode_metrics: RequestSpecDecodeMetrics | None = None

    # [num_scored_rows, num_token_ids]; appended last for array_like compatibility.
    prompt_token_id_logprobs: torch.Tensor | None = None

    # Sent with FinishReason.ERROR, e.g. when a paused engine rejects a request.
    error: RequestError | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class UtilityOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    call_id: int

    # Non-None implies the call failed, result should be None.
    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """后端 EngineCore 每执行完一个 step 后，按 client_index 分组发回前端 OutputProcessor 的批量增量结果。

    注意：这里只传轻量级 token ID 整数（不含解码文本），文本解码留在前端进程异步完成。
    一次混合 Batch Step 产出的典型实例长这样（Sample）：
    ```python
    EngineCoreOutputs(
        engine_index=0,                            # DP 模式下标识来自哪个 EngineCore rank
        outputs=[
            # 1) 普通 Decode 请求：本步正常生成 1 个新 token，尚未结束
            EngineCoreOutput(
                request_id="req_A",
                new_token_ids=[912],
                finish_reason=None,
            ),
            # 2) 开启投机解码 (Speculative Decoding) 的请求：单步同时接受了草稿 token 502 并采样出 EOS 128009，直接完结
            EngineCoreOutput(
                request_id="req_B",
                new_token_ids=[502, 128009],
                finish_reason=FinishReason.STOP,   # 枚举值 0 ("stop")
                stop_reason=128009,
            ),
            # 注：同批中若还有一条处于 Chunked Prefill 中间分片的 "req_C"（Prompt 尚未算完，无新生成 token），
            # 则根本不会出现在 outputs 列表中，从而节省 IPC 带宽与前端遍历开销！
        ],
        finished_requests={"req_B"},
        scheduler_stats=SchedulerStats(num_running_reqs=2, num_waiting_reqs=0, kv_cache_usage=0.34, ...),
    )
    ```
    """

    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0

    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None

    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    # Weight version used for this step.
    weight_version: str | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    new_data_parallel_master_port_list: list[int]
    coord_store_port: int


class ReconfigureRankType(enum.IntEnum):
    """Rank type for reconfiguring distributed request."""

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2


class EngineStatusType(enum.IntEnum):
    HEALTHY = 0
    DEAD = 1
    UNHEALTHY = 2
