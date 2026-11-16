# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.multimodal.utils import strip_covered_mm_data

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.aux_output_connector.connector import (
        AuxOutputConnectorMetadata,
    )
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.request import Request
else:
    AuxOutputConnectorMetadata = object
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    """首次被调度（或 V2 下抢占后重新恢复）的请求完整元数据，发往 Worker 用于注册常驻槽位。

    特别注意 `block_ids: tuple[list[int], ...]` 为什么是外层 tuple 套内层 list：
    - 外层 tuple 的下标对应 **KV Cache Group**（例如标准 Llama/Qwen 只有 1 个全注意力组，外层长度为 1；
      而 Gemma-3 / Jamba 等混合架构同时含 Full Attention 组与 Sliding Window / Mamba 组，外层长度为 2+）；
    - 内层 `list[int]` 才是该请求在该 KV Cache Group 下按序分配到的物理块号列表。

    典型实例（Sample，假设 block_size=16，Prompt 长 40，前 16 个 token 命中 Prefix Cache 块 3，本步先做 8 个 token 的 Chunked Prefill 新分了块 19）：
    ```python
    NewRequestData(
        req_id="req_C",
        prompt_token_ids=[128000, 791, ..., 432],  # 长度 40 的原始 Prompt
        prefill_token_ids=[128000, 791, ..., 432], # V2 专用：正常等于 prompt_token_ids；若曾生成过 5 个 token 后被抢占重算，则长度为 45 (prompt + 已生成 output)
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.7, max_tokens=256),
        pooling_params=None,
        block_ids=([3, 19],),                      # 单 KV 组模型为 1 元 tuple；双组混合模型形如 ([3, 19], [7, 22])
        num_computed_tokens=16,                    # 前 16 个 token (块 3) 已命中缓存无需重算，Worker 将从 index=16 开始算
        lora_request=None,
    )
    ```
    """

    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None
    # DeepSeek-V4.1 only: SWA bounded replay; see Request.replay_start.
    replay_start: int = 0

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
        uses_mrope: bool = False,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=strip_covered_mm_data(
                request.mm_features,
                request.num_computed_tokens,
                uses_mrope=uses_mrope,
            ),
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prompt_is_token_ids=request.prompt_is_token_ids,
            prefill_token_ids=prefill_token_ids,
            replay_start=request.replay_start,
        )

    @property
    def prompt_len(self) -> int:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        if self.prompt_embeds is not None:
            return self.prompt_embeds.shape[0]
        return 0

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    """已在 Worker 显存中建好档案的老请求增量更新包（采用 Struct-of-Arrays 列式存储，最小化序列化体积）。

    无需再传庞大的 prompt_token_ids 或 SamplingParams！同下标 `i` 对应同一条老请求：
    典型实例（Sample，包含两条正在 Decode 的老请求 `req_A` 与 `req_B`，block_size=16）：
    ```python
    CachedRequestData(
        req_ids=["req_A", "req_B"],
        resumed_req_ids=set(),
        new_token_ids=[],                          # 非 PP 模式恒为空列表（Worker 自己知道上一步刚采样出了什么 token！）
        all_token_ids={},
        new_block_ids=[
            None,                                  # req_A: num_computed=21，第 22 个 token 仍装得进已有的第 2 个物理块 (16..31)，无需新块！
            ([25],),                               # req_B: 刚跨过 16 的整数倍边界，本步为第 0 个 KV Cache Group 新追加了物理块 25
        ],
        num_computed_tokens=[21, 16],              # 调度本步前各自已完成的 KV token 数（含对上一步投机解码拒绝回滚后的修正值）
        num_output_tokens=[2, 5],                  # 已生成的 output token 数（等于 0 说明仍处于长 Prompt 的后续 Chunked Prefill 分片）
    )
    ```
    """

    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # MRV1-only: For requests not scheduled in the last step, propagate the token ids
    # to the connector. Won't contain requests scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class ScheduledEncoderInputStats:
    """Stats for encoder inputs scheduled in one iteration."""

    num_inputs: int = 0
    output_tokens: int = 0


@dataclass
class KVConnectorBlockState:
    """Scheduler-local block state offered to a producer-side KV connector."""

    # Requests scheduled this step and requests with a boundary-state hand-off.
    req_ids: set[str]
    # Resolve on access to avoid copying tables the connector never reads.
    resolve_block_ids: Callable[[str], tuple[list[int], ...]]
    # Exact Mamba "align" boundary-state hand-offs.
    boundary_state_offloads: dict[str, list[tuple[int, int, int]]]

    def get_block_ids(self, req_id: str) -> tuple[list[int], ...] | None:
        if req_id not in self.req_ids:
            return None
        return self.resolve_block_ids(req_id)


@dataclass
class SchedulerOutput:
    """Scheduler.schedule() 每步产出的核心调度指令包，经 Executor 广播给所有 GPU Worker。

    一次典型的 Continuous Batching 混合批次实例长这样（Sample）：
    假设本步同时包含：
      - `req_A`：普通 Decode 老请求（算 1 个新 token）
      - `req_B`：带投机解码的 Decode 老请求（1 个上步采样 token + 2 个草稿 token = 共算 3 个 token）
      - `req_C`：首次准入的新请求（Prompt 长 40，命中缓存 16，本步分片 Prefill 算 8 个 token）
      - `req_prev`：上一步刚遇到 EOS 结束的请求，本步顺带通知 Worker 释放其槽位
    ```python
    SchedulerOutput(
        scheduled_new_reqs=[NewRequestData(req_id="req_C", num_computed_tokens=16, block_ids=([3, 19],), ...)],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["req_A", "req_B"],
            new_block_ids=[None, ([25],)],
            num_computed_tokens=[21, 10],
            num_output_tokens=[2, 5], ...
        ),
        num_scheduled_tokens={                     # 每条请求本步实际送入模型 forward 的 token 数量
            "req_A": 1,
            "req_B": 3,                            # 1 (bonus/target) + 2 (drafts)
            "req_C": 8,                            # Chunked Prefill 本步分片大小
        },
        total_num_scheduled_tokens=12,             # 1 + 3 + 8 = 12（决定 GPU 扁平化 input_ids 的总长度）
        scheduled_spec_decode_tokens={
            "req_B": [502, 503],                   # 仅含带草稿 token 的请求；普通请求不入字典
        },
        scheduled_encoder_inputs={},               # 若 req_C 含第 0 张图片需在本步跑 ViT，则为 {"req_C": [0]}
        finished_req_ids={"req_prev"},             # 通知 Worker 从 RequestState 槽位表中清理 req_prev
        preempted_req_ids=set(),                   # 本步因显存不足被抢占踢回 waiting 的请求 ID
        new_block_ids_to_zero=None,                # 混合/FP8 KV 模型需要预先清零的新分配物理块 ID 列表
    )
    ```
    """

    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request-local CPU payload updates: None releases a consumed input; a
    # payload restores it for re-encoding after KV-load rollback. GPU embeddings
    # and the feature's identifier/placeholder metadata are unaffected.
    encoder_input_updates: dict[str, dict[int, "MultiModalKwargsItem | None"]] = field(
        default_factory=dict
    )

    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # Whether any scheduled request consumes KV that the connector loads
    # synchronously during this step (load_async=False).
    has_sync_kv_loads: bool = False

    # Execution-auxiliary output control metadata consumed by the worker connector.
    aux_output_connector_metadata: AuxOutputConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None
    # EC Cache Manager metadata
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Scheduler-local; always None by the time this reaches a worker.
    kv_connector_block_state: KVConnectorBlockState | None = None

    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0

    # Deprecated: cascade attention has been removed, so this is always empty.
    # Kept for out-of-tree schedulers and workers.
    num_common_prefix_blocks: list[int] = field(default_factory=list)

    # Whether no scheduled request can use the drafts proposed after this step
    # (non-final prefill chunks, or requests sure to reach max_tokens), so the
    # drafter can skip proposing them.
    skip_drafts: bool = False

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    """结构化输出（JSON Schema / Regex / EBNF）的词表位图掩码载荷，在 sample_tokens 阶段送入 GPU。

    典型实例（Sample，假设词表大小 vocab_size=128000，每 32 个 token 压入 1 个 int32，即每行 128000 / 32 = 4000 个 int32）：
    ```python
    GrammarOutput(
        structured_output_request_ids=["req_A"],
        # shape = [num_grammar_reqs * mask_stride, ceil(vocab_size / 32)]，例如 (1, 4000), dtype=int32
        # 其中第 j 个 bit 为 1 表示允许采样对应 token，为 0 表示非法（将在 GPU 上被置为 -inf）
        grammar_bitmask=array([[0b000...1010, ...]], dtype=int32),
        num_acceptable_drafts=[0],
    )
    ```
    """

    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
    # Per request, ordered as structured_output_request_ids: how many leading
    # drafts the bitmask constrained. `grammar_bitmask` fills every row after
    # the first -1 placeholder with the all-permissive `_full_mask`, so drafts
    # from this index on must not be accepted. None invalidates every draft.
    num_acceptable_drafts: list[int] | None = None
