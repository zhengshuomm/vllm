# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import itertools
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm
from typing_extensions import TypeAlias

import vllm.envs as envs
from vllm.attention import Attention, AttentionType
from vllm.attention.backends.abstract import AttentionBackend, MultipleOf
from vllm.attention.layer import MLAAttention
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationLevel,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    DeviceMemoryProfiler,
    GiB_bytes,
    cdiv,
    check_use_alibi,
    get_dtype_size,
    is_pin_memory_available,
    length_from_prompt_token_ids_or_embeds,
    round_up,
    supports_dynamo,
)
from vllm.utils.jsontree import json_map_leaves
from vllm.v1.attention.backends.flash_attn import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    create_fast_prefill_custom_backend,
    reorder_batch_to_split_decodes_and_prefills,
    split_attn_metadata,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .utils import (
    AttentionGroup,
    MultiModalBudget,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    gather_mm_placeholders,
    sanity_check_mm_encoder_outputs,
    scatter_mm_placeholders,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[list[AttnMetadataDict], AttnMetadataDict]


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self._async_copy_ready_event = torch.cuda.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        self._async_copy_ready_event.synchronize()

        # Release the device tensor once the copy has completed
        del self._sampled_token_ids

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))
        from vllm.model_executor.layers.batch_invariant import init_batch_invariance

        init_batch_invariance()

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 0
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )  # type: ignore
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors,
            ),
            is_pooling_model=self.is_pooling_model,
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.async_output_copy_stream = (
            torch.cuda.Stream() if self.use_async_scheduling else None
        )

        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = list(
                reversed(self.compilation_config.cudagraph_capture_sizes)
            )

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_indices = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )
        self.num_discarded_requests = 0

        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            self.is_mm_embed = self._make_buffer(self.max_num_tokens, dtype=torch.bool)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # CUDA event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: Optional[torch.cuda.Event] = None
        if self.use_async_scheduling:
            self.prepare_inputs_event = torch.cuda.Event()
            # Start in a completed state.
            self.prepare_inputs_event.record(torch.cuda.default_stream())

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = (
            1
            if not self.speculative_config
            else 1 + self.speculative_config.num_speculative_tokens
        )

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            )
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: Optional[int] = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]], torch.Tensor]] = None
        self.transfer_event = torch.cuda.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

    def _get_positions(self, num_tokens: Any):
        """
        获取位置编码张量
        
        根据token数量获取相应的位置编码，支持M-RoPE和普通位置编码
        
        Args:
            num_tokens: token数量，可以是整数或张量索引
            
        Returns:
            torch.Tensor: 位置编码张量
        """
        # ==================== 处理整数索引 ====================
        if isinstance(num_tokens, int):  # 如果是整数索引
            if self.uses_mrope:  # 如果使用M-RoPE
                return self.mrope_positions.gpu[:, :num_tokens]  # 返回M-RoPE位置编码
            return self.positions.gpu[:num_tokens]  # 返回普通位置编码
        else:
            # ==================== 处理张量索引 ====================
            if self.uses_mrope:  # 如果使用M-RoPE
                return self.mrope_positions.gpu[:, num_tokens]  # 返回M-RoPE位置编码
            return self.positions.gpu[num_tokens]  # 返回普通位置编码

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        """
        创建CPU-GPU缓冲区
        
        创建一个可以在CPU和GPU之间高效传输数据的缓冲区，
        用于存储张量数据
        
        Args:
            *size: 缓冲区尺寸参数
            dtype: 数据类型
            numpy: 是否支持numpy操作
            
        Returns:
            CpuGpuBuffer: CPU-GPU缓冲区实例
        """
        # ==================== 创建缓冲区 ====================
        return CpuGpuBuffer(
            *size,                    # 缓冲区尺寸
            dtype=dtype,              # 数据类型
            device=self.device,       # 设备
            pin_memory=self.pin_memory,  # 是否固定内存
            with_numpy=numpy,         # 是否支持numpy
        )

    def _init_model_kwargs(self, num_tokens: int):
        """
        初始化模型关键字参数
        
        为模型前向传播准备必要的参数，特别是池化模型需要的
        token_type_ids等特殊参数
        
        Args:
            num_tokens: token数量
            
        Returns:
            dict[str, Any]: 模型关键字参数字典
        """
        # ==================== 初始化参数字典 ====================
        model_kwargs = dict[str, Any]()  # 创建空的参数字典

        # ==================== 检查是否为池化模型 ====================
        if not self.is_pooling_model:
            return model_kwargs  # 如果不是池化模型，返回空字典

        # ==================== 获取池化参数 ====================
        num_reqs = self.input_batch.num_reqs  # 获取请求数量
        pooling_params = self.input_batch.get_pooling_params()  # 获取池化参数

        # ==================== 处理token类型ID ====================
        # 收集需要特殊token类型ID的请求
        token_type_id_requests = dict[int, Any]()  # 存储token类型ID请求
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None  # 有额外参数
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))  # 获取压缩的token类型ID
                is not None
            ):
                token_type_id_requests[i] = token_types  # 存储请求的token类型ID

        # ==================== 检查是否有token类型ID请求 ====================
        if len(token_type_id_requests) == 0:
            return model_kwargs  # 如果没有特殊请求，返回空字典

        # ==================== 生成token类型ID ====================
        seq_lens = self.seq_lens.gpu[:num_reqs]  # 获取序列长度（GPU）
        token_type_ids = []  # 存储token类型ID列表

        for i in range(num_reqs):  # 遍历每个请求
            pos = token_type_id_requests.get(i, seq_lens[i])  # 获取分割位置
            # 生成token类型ID：位置之前为0，位置之后为1
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)  # 添加到列表

        # ==================== 设置模型参数 ====================
        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        根据注意力后端的需求更新批次中请求的顺序
        
        某些注意力后端（如MLA）可能希望根据注意力计算是计算受限
        还是内存受限来分离请求，以优化性能

        Args:
            scheduler_output: 调度器输出
        """
        # ==================== 检查注意力模型类型 ====================
        # 无注意力模型有零个kv_cache_groups，但是像Mamba这样的模型
        # 也是无注意力的，但使用kv_cache来保持其内部状态
        # 这就是为什么我们检查kv_cache_groups的数量而不是仅仅检查
        # self.model_config.is_attention_free
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return  # 如果是无注意力模型，不需要重排序

        # ==================== 检查重排序阈值 ====================
        if self.reorder_batch_threshold is not None:
            # 注意：目前没有后端支持自定义掩码
            #  required for DCP with q_len > 1, so we assert here. Remove this
            #  assert once the custom mask is support is added to FA3.
            if (
                self.dcp_world_size > 1
                and envs.VLLM_ATTENTION_BACKEND != "FLASH_ATTN_MLA"
            ):
                assert self.reorder_batch_threshold == 1, (
                    "DCP not support reorder_batch_threshold > 1 now."
                )
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """
        初始化设备属性
        
        从torch.cuda.get_device_properties获取GPU设备属性，
        用于后续的性能优化和资源管理
        """
        # ==================== 获取设备属性 ====================
        self.device_properties = torch.cuda.get_device_properties(self.device)  # 获取GPU设备属性
        
        # ==================== 设置SM数量 ====================
        self.num_sms = self.device_properties.multi_processor_count  # 获取流多处理器数量

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        """
        同步设备操作
        
        等待所有CUDA操作完成，确保GPU计算同步
        用于模型运行器重写
        """
        # ==================== 同步CUDA操作 ====================
        torch.cuda.synchronize()  # 等待所有CUDA操作完成

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """
        更新缓存状态和持久化批次，根据调度器输出
        
        更新的状态被 `_prepare_inputs` 函数用来创建模型的输入GPU张量。
        如果批次中有新的/恢复的/暂停的/完成的请求，SamplingMetadata会被更新并复制到GPU。
        """
        
        # ==================== 清理已完成的请求 ====================
        # 从缓存状态中移除已完成的请求
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        
        # 从持久化批次中移除已完成的请求
        # 注意：可能存在finished_req_ids和scheduled_req_ids重叠的边缘情况
        # 这发生在请求被中止然后以相同ID重新提交时
        # 在这种情况下，我们将它们视为两个不同的请求
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # ==================== 释放编码器缓存 ====================
        # 释放缓存的编码器输出（多模态数据）
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # ==================== 移除未调度的请求 ====================
        # 从持久化批次中移除未调度的请求
        # 注意：未调度的请求要么是被抢占的请求，要么是在此步骤中未调度的运行中请求
        # 我们从持久化批次中移除它们但保留其缓存状态，因为它们将来会被重新调度
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()  # 获取此步骤调度的请求ID
        cached_req_ids = self.input_batch.req_id_to_index.keys()         # 获取当前缓存的请求ID
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids          # 计算未调度的请求ID
        
        # 注意：持久化批次优化假设连续批次包含大部分相同的请求
        # 如果批次请求重叠度低（例如，在两个不同请求集之间交替），此优化会变得非常低效
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        # ==================== 添加新请求 ====================
        reqs_to_add: list[CachedRequestState] = []  # 准备添加的请求列表
        
        # 将新请求添加到缓存状态
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id                    # 获取请求ID
            sampling_params = new_req_data.sampling_params  # 获取采样参数
            pooling_params = new_req_data.pooling_params    # 获取池化参数

            # 处理随机种子生成器
            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                # 创建指定种子的随机数生成器
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            # 处理池化模型的参数更新
            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                # 获取池化器并应用参数更新
                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            # 创建新的缓存请求状态
            req_state = CachedRequestState(
                req_id=req_id,                                    # 请求ID
                prompt_token_ids=new_req_data.prompt_token_ids,   # 提示token IDs
                prompt_embeds=new_req_data.prompt_embeds,         # 提示嵌入
                mm_features=new_req_data.mm_features,             # 多模态特征
                sampling_params=sampling_params,                  # 采样参数
                pooling_params=pooling_params,                    # 池化参数
                generator=generator,                              # 随机数生成器
                block_ids=new_req_data.block_ids,                # 块ID
                num_computed_tokens=new_req_data.num_computed_tokens,  # 已计算token数
                output_token_ids=[],                              # 输出token IDs
                lora_request=new_req_data.lora_request,           # LoRA请求
            )
            self.requests[req_id] = req_state  # 将请求状态添加到缓存

            # 仅对使用M-RoPE的模型相关（例如Qwen2-VL）
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)  # 添加到待添加列表

        # ==================== 更新运行中/恢复的请求 ====================
        is_last_rank = get_pp_group().is_last_rank  # 检查是否为流水线并行的最后一个rank
        req_data = scheduler_output.scheduled_cached_reqs  # 获取调度的缓存请求数据
        
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]  # 获取请求状态
            num_computed_tokens = req_data.num_computed_tokens[i]      # 已计算token数
            new_block_ids = req_data.new_block_ids[i]                  # 新块ID
            resumed_from_preemption = req_data.resumed_from_preemption[i]  # 是否从抢占恢复
            num_output_tokens = req_data.num_output_tokens[i]          # 输出token数

            # 更新缓存状态
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]

                req_index = self.input_batch.req_id_to_index.get(req_id)
                if req_index is not None:
                    old_end_idx = self.input_batch.num_tokens_no_spec[req_index]
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens[req_index] = end_idx
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx
                    self.input_batch.is_token_ids[req_index, end_idx:old_end_idx] = (
                        False
                    )

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ()
            )
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index
                ] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens
                self.input_batch.spec_token_ids[req_index] = spec_token_ids

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor
    ) -> None:
        """
        模型执行后更新缓存状态
        
        用于MTP/EAGLE混合模型，因为在线性注意力中，
        只保留最后一个token的状态。在MTP/EAGLE中，对于草稿token，
        状态会一直保持到我们决定每个序列接受多少个token，
        然后在下一次迭代中根据接受的token数量进行状态偏移。
        
        Args:
            output_token_ids: 输出的token IDs张量
        """
        # ==================== 检查模型类型 ====================
        if not self.model_config.is_hybrid or not self.speculative_config:
            return  # 如果不是混合模型或没有推测配置，直接返回

        # ==================== 计算接受的token数量 ====================
        # 找到每个序列接受的token数量
        num_accepted_tokens = (
            (
                torch.cat(
                    [
                        output_token_ids,  # 输出token IDs
                        torch.full(
                            (output_token_ids.size(0), 1),  # 创建填充张量
                            -1,  # 填充值为-1
                            device=output_token_ids.device,  # 相同设备
                        ),
                    ],
                    dim=1,  # 在维度1上连接
                )
                == -1  # 检查哪些位置是填充值
            )
            .int()
            .argmax(-1)
            .cpu()
            .numpy()
        )
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _init_mrope_positions(self, req_state: CachedRequestState):
        """
        初始化M-RoPE位置编码
        
        为多模态请求初始化M-RoPE（Multi-modal Rotary Position Embedding）位置编码，
        处理图像、视频、音频等多模态数据的位置信息
        
        Args:
            req_state: 缓存的请求状态
        """
        # ==================== 初始化多模态参数 ====================
        image_grid_thw = []  # 图像网格尺寸 (time, height, width)
        video_grid_thw = []  # 视频网格尺寸 (time, height, width)
        second_per_grid_ts = []  # 每个网格的秒数
        audio_feature_lengths = []  # 音频特征长度
        use_audio_in_video = False  # 是否在视频中使用音频

        # ==================== 处理多模态特征 ====================
        for mm_feature in req_state.mm_features:  # 遍历多模态特征
            mm_item = mm_feature.data  # 获取多模态数据项
            if mm_item is None:
                continue  # 如果数据为空，跳过
            
            mm_input = mm_item.get_data()  # 获取多模态输入数据
            
            # 提取图像网格尺寸
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            
            # 提取视频网格尺寸
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            
            # 提取每个网格的秒数
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            
            # 提取音频特征长度
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            
            # 检查是否在视频中使用音频
            if mm_input.get("use_audio_in_video") is True:
                use_audio_in_video = True

        # ==================== 生成M-RoPE位置编码 ====================
        if supports_mrope(self.model):  # 如果模型支持M-RoPE
            req_state.mrope_positions, req_state.mrope_position_delta = (
                self.model.get_mrope_input_positions(
                    req_state.prompt_token_ids,  # 提示token IDs
                    hf_config=self.model_config.hf_config,  # HuggingFace配置
                    image_grid_thw=image_grid_thw,  # 图像网格尺寸
                    video_grid_thw=video_grid_thw,  # 视频网格尺寸
                    second_per_grid_ts=second_per_grid_ts,  # 每个网格的秒数
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )
            )
        else:
            req_state.mrope_positions, req_state.mrope_position_delta = (
                MRotaryEmbedding.get_input_positions_tensor(
                    req_state.prompt_token_ids,
                    hf_config=self.model_config.hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )
            )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        """
        提取多模态关键字参数
        
        从调度器输出中提取多模态数据，组织成模型可以处理的格式
        
        Args:
            scheduler_output: 调度器输出，包含多模态请求信息
            
        Returns:
            BatchedTensorInputs: 批处理的多模态张量输入
        """
        # ==================== 检查条件 ====================
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}  # 如果没有调度器输出或不是多模态原始输入模型，返回空字典

        # ==================== 收集多模态参数 ====================
        mm_kwargs = list[MultiModalKwargsItem]()  # 创建多模态参数列表
        for req in scheduler_output.scheduled_new_reqs:  # 遍历新调度的请求
            for feature in req.mm_features:  # 遍历多模态特征
                if feature.data is not None:  # 如果特征数据不为空
                    mm_kwargs.append(feature.data)  # 添加到参数列表

        # ==================== 按模态分组处理 ====================
        # 一次性输入所有模态
        model = cast(SupportsMultiModal, self.model)  # 转换为支持多模态的模型
        mm_kwargs_combined: BatchedTensorInputs = {}  # 组合的多模态参数
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,  # 多模态参数
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        """
        创建虚拟多模态关键字参数
        
        用于测试或预热时创建虚拟的多模态输入数据
        
        Args:
            num_seqs: 序列数量
            
        Returns:
            BatchedTensorInputs: 虚拟的多模态张量输入
        """
        # ==================== 检查模型类型 ====================
        if not self.is_multimodal_raw_input_only_model:
            return {}  # 如果不是多模态原始输入模型，返回空字典

        # ==================== 获取多模态预算 ====================
        mm_budget = self.mm_budget  # 获取多模态预算
        assert mm_budget is not None  # 确保预算不为空

        # ==================== 选择虚拟模态 ====================
        dummy_modality = mm_budget.get_modality_with_max_tokens()  # 获取token数量最多的模态
        
        # ==================== 创建虚拟批次 ====================
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)  # 创建虚拟多模态批次

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        获取累积和和批处理范围数组
        
        计算给定数组的累积和和批处理的arange，用于生成位置索引
        例如：[2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        等价于但比 np.concatenate([np.arange(n) for n in num_tokens]) 更快
        
        Args:
            num_tokens: token数量数组
            cumsum_dtype: 累积和的数据类型
            
        Returns:
            tuple: 包含累积和和arange的元组
        """
        # ==================== 步骤1：计算累积和 ====================
        # [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)  # 计算累积和
        total_num_tokens = cu_num_tokens[-1]  # 获取总token数量
        
        # ==================== 步骤2：计算偏移量 ====================
        # [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)  # 重复偏移量
        
        # ==================== 步骤3：生成arange ====================
        # [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets  # 生成范围数组

        return cu_num_tokens, arange  # 返回累积和和arange

    def _prepare_input_ids(
        self, total_num_scheduled_tokens: int, cu_num_tokens: np.ndarray
    ) -> None:
        """
        准备当前批次的输入ID

        仔细处理 `prev_sampled_token_ids`，这些token可能从之前的引擎迭代中缓存，
        在这种情况下，需要将GPU上的这些token复制到input_ids中的相应位置

        Args:
            total_num_scheduled_tokens: 总调度的token数量
            cu_num_tokens: 累积token数量数组
        """
        # ==================== 正常调度情况 ====================
        if self.input_batch.prev_sampled_token_ids is None:
            # 正常调度情况
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)  # 复制input_ids到GPU
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)  # 复制嵌入到GPU
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)  # 复制token ID标志到GPU
            return

        # ==================== 异步调度情况 ====================
        # 异步调度情况，其中来自前一次迭代的一些解码请求
        # 在input_ids_cpu中没有条目，需要从prev_sampled_token_ids复制到GPU
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index  # 获取之前的请求ID到索引映射
        assert prev_req_id_to_index is not None  # 确保映射不为空
        
        # ==================== 初始化变量 ====================
        flattened_indices = []  # 展平的索引列表
        prev_common_req_indices = []  # 之前的公共请求索引
        indices_match = True  # 索引是否匹配
        max_flattened_index = -1  # 最大展平索引
        
        # ==================== 处理请求索引 ====================
        for req_id, cur_index in self.input_batch.req_id_to_index.items():  # 遍历当前请求索引
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:  # 如果请求在之前也存在
                prev_common_req_indices.append(prev_index)  # 添加到公共索引列表
                # 我们需要计算展平的input_ids索引
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                flattened_indices.append(flattened_index)
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        # ==================== 计算公共token数量 ====================
        num_commmon_tokens = len(flattened_indices)  # 计算公共token数量
        
        # ==================== 处理部分请求情况 ====================
        if num_commmon_tokens < total_num_scheduled_tokens:
            # 如果不是所有请求都来自上次迭代的解码，
            # 我们需要先将input_ids_cpu复制到GPU
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)  # 复制input_ids到GPU
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)  # 复制嵌入到GPU
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)  # 复制token ID标志到GPU
        
        # ==================== 处理无公共请求情况 ====================
        if num_commmon_tokens == 0:
            # 与上次迭代没有公共请求
            # 所以input_ids_cpu将包含所有输入ID
            return
        
        # ==================== 优化：直接复制情况 ====================
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # 常见情况优化：批次未改变且没有发生重排序
            # 索引都是0..N-1的相同排列，所以我们可以使用单个切片直接复制
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],  # 复制之前的采样token IDs
                non_blocking=True,  # 非阻塞复制
            )
            if self.enable_prompt_embeds:  # 如果启用了提示嵌入
                self.is_token_ids.gpu[:num_commmon_tokens] = True  # 设置token ID标志
            return
        
        # ==================== 异步上传索引张量 ====================
        # 异步上传索引张量，使scatter操作可以非阻塞
        input_ids_index_tensor = torch.tensor(
            flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory  # 创建展平索引张量
        ).to(self.device, non_blocking=True)  # 异步传输到设备
        
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory  # 创建之前公共请求索引张量
        ).to(self.device, non_blocking=True)  # 异步传输到设备
        
        # ==================== 执行scatter操作 ====================
        self.input_ids.gpu.scatter_(
            dim=0,  # 在维度0上scatter
            index=input_ids_index_tensor,  # 目标索引
            src=self.input_batch.prev_sampled_token_ids[  # 源数据
                prev_common_req_indices_tensor, 0  # 从之前的采样token IDs中获取
            ],
        )

    def _get_encoder_seq_lens(
        self,
        scheduler_output: "SchedulerOutput",
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
    ) -> Optional[np.ndarray]:
        """
        获取编码器序列长度
        
        为编码器-解码器模型构建编码器序列长度数组，
        映射请求索引到此批次中调度的输入的编码器长度
        
        Args:
            scheduler_output: 调度器输出
            kv_cache_spec: KV缓存规格
            num_reqs: 请求数量
            
        Returns:
            Optional[np.ndarray]: 编码器序列长度数组，如果不是交叉注意力规格则返回None
        """
        # ==================== 检查KV缓存规格 ====================
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None  # 如果不是交叉注意力规格，返回None

        # ==================== 构建编码器序列长度数组 ====================
        # 构建encoder_seq_lens数组，将请求索引映射到
        # 此批次中调度的输入的编码器长度
        encoder_seq_lens = np.zeros(num_reqs, dtype=np.int32)  # 初始化编码器序列长度数组
        
        # ==================== 填充编码器长度 ====================
        for req_id in scheduler_output.scheduled_encoder_inputs:  # 遍历调度的编码器输入
            req_index = self.input_batch.req_id_to_index[req_id]  # 获取请求索引
            encoder_seq_lens[req_index] = self.max_encoder_len  # 设置最大编码器长度

        return encoder_seq_lens  # 返回编码器序列长度数组

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[
        PerLayerAttnMetadata,
        torch.Tensor,
        Optional[SpecDecodeMetadata],
        np.ndarray,
        Optional[CommonAttentionMetadata],
        int,
        Optional[UBatchSlices],
        Optional[torch.Tensor],
        bool,
    ]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            logits_indices, spec_decode_metadata,
            num_scheduled_tokens, spec_decode_common_attn_metadata,
            max_num_scheduled_tokens, use_cascade_attn
        ]
        """
        # ==================== 基本验证和初始化 ====================
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取总调度的token数量
        assert total_num_scheduled_tokens > 0  # 确保有token需要处理
        num_reqs = self.input_batch.num_reqs  # 获取请求数量
        assert num_reqs > 0  # 确保有请求需要处理

        # ==================== 优化：首先复制块表 ====================
        # 这样我们可以将复制操作与后续的CPU操作重叠，提高性能
        self.input_batch.block_table.commit_block_table(num_reqs)

        # ==================== 获取每个请求的调度token数量 ====================
        req_ids = self.input_batch.req_ids  # 获取请求ID列表
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]  # 获取每个请求的token数量
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)  # 转换为numpy数组
        max_num_scheduled_tokens = max(tokens)  # 计算最大token数量

        # ==================== 生成请求索引 ====================
        # 例如：[2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # 每个请求的token数量对应重复的请求索引
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # ==================== 计算累积token数和范围 ====================
        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]  # 累积token数量
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]  # 每个请求内的token位置
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # ==================== 计算位置编码 ====================
        positions_np = self.positions.np[:total_num_scheduled_tokens]  # 获取位置数组
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],  # 已计算的token数（每个请求的起始位置）
            arange,                                                 # 当前批次中的位置（0, 1, 2, ...）
            out=positions_np,                                       # 输出到位置数组
        )

        # ==================== 计算M-RoPE位置 ====================
        # 仅对使用M-RoPE的模型相关（例如Qwen2-VL）
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # ==================== 计算token索引 ====================
        # 例如：[0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # 其中M是max_model_len，用于在2D张量中定位token
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)  # 转换为PyTorch张量

        # ==================== 提取token IDs ====================
        # 注意：我们使用torch.index_select而不是np.take
        # 因为torch.index_select对于大张量来说比np.take快得多
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),  # 展平的token IDs张量
            0,                                                # 维度
            token_indices_tensor,                             # 索引张量
            out=self.input_ids.cpu[:total_num_scheduled_tokens],  # 输出到input_ids
        )
        
        # ==================== 处理提示嵌入标志 ====================
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids.flatten()  # 获取是否为token IDs的标志
            torch.index_select(
                is_token_ids,                                    # 标志张量
                0,                                              # 维度
                token_indices_tensor,                           # 索引张量
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],  # 输出
            )

        # ==================== 处理提示嵌入 ====================
        # 因为我们没有在InputBatch上预分配大量的prompt_embeds CPU张量
        # 我们需要将提示嵌入填充到GpuModelRunner预分配的prompt_embeds张量的预期位置
        if self.input_batch.req_prompt_embeds:
            output_idx = 0  # 输出索引
            for req_idx in range(num_reqs):  # 遍历每个请求
                num_sched = num_scheduled_tokens[req_idx]  # 获取此请求调度的token数量

                # 如果此请求没有嵌入，跳过
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # 如果没有调度的token，跳过
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]  # 获取请求的嵌入
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]  # 获取起始位置

                # ==================== 检查嵌入边界 ====================
                # 如果尝试读取超出可用嵌入的范围，跳过
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # ==================== 复制可用嵌入 ====================
                end_pos = start_pos + num_sched  # 计算结束位置
                actual_end = min(end_pos, req_embeds.shape[0])  # 实际结束位置（不超过嵌入长度）
                actual_num_sched = actual_end - start_pos  # 实际调度的数量

                if actual_num_sched > 0:  # 如果有实际调度的嵌入
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])  # 复制嵌入到CPU缓冲区

                output_idx += num_sched  # 更新输出索引

        # ==================== 计算槽映射 ====================
        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)  # 计算槽映射
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)  # 提交槽映射

        # ==================== 准备注意力元数据 ====================
        self.query_start_loc.np[0] = 0  # 设置查询起始位置为0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens  # 设置每个请求的查询起始位置
        # 注意：填充query_start_loc使其非递减，因为像FlashAttention这样的内核需要这样
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])  # 填充剩余位置
        self.query_start_loc.copy_to_gpu()  # 复制到GPU
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]  # 获取GPU上的查询起始位置

        # ==================== 计算token数量和填充 ====================
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens  # 未填充的token数量
        num_tokens_padded = self._get_num_input_tokens(num_tokens_unpadded)  # 填充后的token数量
        uniform_decode = (
            max_num_scheduled_tokens == self.uniform_decode_query_len
        ) and (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)  # 检查是否为统一解码
        
        # ==================== 协调数据并行批次 ====================
        ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
            num_scheduled_tokens,  # 调度的token数量
            num_tokens_unpadded,   # 未填充的token数量
            num_tokens_padded,     # 填充后的token数量
            self.parallel_config,  # 并行配置
            True,                  # 启用协调
            uniform_decode,        # 是否统一解码
        )

        # ==================== 设置序列长度 ====================
        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens  # 计算序列长度
        )
        # 为完整的CUDA Graph模式填充未使用的部分为0
        self.seq_lens.np[num_reqs:].fill(0)  # 填充未使用的序列长度为0
        self.seq_lens.copy_to_gpu()  # 复制到GPU
        seq_lens = self.seq_lens.gpu[:num_reqs]  # 获取GPU上的序列长度
        max_seq_len = self.seq_lens.np[:num_reqs].max().item()  # 计算最大序列长度

        # ==================== 计算token数量 ====================
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]  # 获取每个请求的token数量
        num_tokens_np = np.array(num_tokens, dtype=np.int32)  # 转换为numpy数组

        # ==================== 记录丢弃的请求 ====================
        # 记录不应该被采样的请求索引，这样我们可以在返回前清除采样的token
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np  # 创建丢弃请求掩码
        discard_request_indices = np.nonzero(discard_requests_mask)[0]  # 获取丢弃请求的索引
        self.num_discarded_requests = len(discard_request_indices)  # 设置丢弃请求数量
        self.discard_request_indices.np[: self.num_discarded_requests] = (
            discard_request_indices  # 设置丢弃请求索引
        )

        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)  # 复制到GPU

        # ==================== 复制张量到GPU ====================
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)  # 准备输入ID

        # ==================== 处理位置编码 ====================
        if self.uses_mrope:  # 如果使用M-RoPE
            # 仅对使用M-RoPE的模型相关（例如Qwen2-VL）
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],  # 复制M-RoPE位置
                non_blocking=True,  # 非阻塞复制
            )
        else:
            # 常见情况（1D位置）
            self.positions.copy_to_gpu(total_num_scheduled_tokens)  # 复制位置到GPU

        # ==================== 处理推测解码 ====================
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0  # 检查是否使用推测解码
        if not use_spec_decode:  # 如果不使用推测解码
            # 注意：由于分块预填充，批次可能包含部分请求。
            # 虽然我们不应该从这些部分请求中采样任何token，但为了简单起见我们这样做。
            # 我们将忽略来自部分请求的采样token。
            # TODO: 支持提示logprobs
            logits_indices = query_start_loc[1:] - 1  # 计算logits索引
            num_draft_tokens = None  # 草稿token数量为None
            spec_decode_metadata = None  # 推测解码元数据为None
        else:
            # ==================== 获取每个请求的草稿token数量 ====================
            # 遍历字典而不是所有请求，因为不是所有请求都有草稿token
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)  # 初始化草稿token数量数组
            # 对于分块预填充，使用-1作为掩码而不是0，因为引导解码可能回滚推测token
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)  # 初始化解码草稿token数量
            
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():  # 遍历推测解码token
                req_idx = self.input_batch.req_id_to_index[req_id]  # 获取请求索引
                num_draft_tokens[req_idx] = len(draft_token_ids)  # 设置草稿token数量
                num_decode_draft_tokens[req_idx] = (
                    len(draft_token_ids)  # 设置解码草稿token数量
                    if (
                        self.input_batch.num_computed_tokens_cpu[req_idx]  # 如果已计算token数
                        >= self.input_batch.num_prompt_tokens[req_idx]  # 大于等于提示token数
                    )
                    else -1  # 否则设为-1
                )
            
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens  # 计算推测解码元数据
            )
            logits_indices = spec_decode_metadata.logits_indices  # 获取logits索引

            # ==================== 处理某些注意力后端的DECODE专用CUDA Graph ====================
            # 例如GDN
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens  # 设置解码草稿token数量
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)  # 填充剩余为-1
            self.num_decode_draft_tokens.copy_to_gpu()  # 复制到GPU

        # ==================== 处理KV共享快速预填充 ====================
        logits_indices_padded = None  # 初始化填充的logits索引
        if self.cache_config.kv_sharing_fast_prefill:  # 如果启用KV共享快速预填充
            logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices  # 准备KV共享快速预填充
            )

        # ==================== 初始化注意力元数据 ====================
        attn_metadata: PerLayerAttnMetadata = {}  # 初始化注意力元数据
        if ubatch_slices is not None:  # 如果有统一批次切片
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]  # 为每个切片创建字典
        use_cascade_attn = False  # 初始化级联注意力标志

        # ==================== 准备循环中使用的变量 ====================
        query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs + 1]  # CPU上的查询起始位置
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]  # CPU上的序列长度
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs
        ]  # CPU上已计算的token数量
        spec_decode_common_attn_metadata = None  # 推测解码通用注意力元数据
        
        if use_spec_decode:  # 如果使用推测解码
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]  # 设置接受的token数量
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)  # 填充剩余为1
            self.num_accepted_tokens.copy_to_gpu()  # 复制到GPU

        # ==================== 为每个KV缓存组准备注意力元数据 ====================
        # 使同一组中的层共享相同的元数据
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups  # 遍历KV缓存组
        ):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs  # 获取编码器序列长度
            )

            # ==================== 处理编码器专用层 ====================
            if isinstance(kv_cache_group_spec.kv_cache_spec, EncoderOnlyAttentionSpec):
                # 编码器专用层没有KV缓存，所以我们需要为它们创建虚拟块表和槽映射
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),  # 创建虚拟块表张量
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens,),  # 创建虚拟槽映射
                    dtype=torch.int64,
                    device=self.device,
                )
                num_common_prefix_blocks = 0  # 公共前缀块数为0
            else:
                # ==================== 处理普通KV缓存 ====================
                blk_table = self.input_batch.block_table[kv_cache_group_id]  # 获取块表
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)  # 获取设备张量
                slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]  # 获取槽映射

                # 为完整的CUDA Graph模式填充未使用的部分为-1
                # 这是reshape_and_cache所需要的
                blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(-1)  # 填充未使用部分
                num_common_prefix_blocks = scheduler_output.num_common_prefix_blocks[
                    kv_cache_group_id
                ]  # 获取公共前缀块数

            # ==================== 创建通用注意力元数据 ====================
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,  # 查询起始位置
                query_start_loc_cpu=query_start_loc_cpu,  # CPU上的查询起始位置
                seq_lens=seq_lens,  # 序列长度
                seq_lens_cpu=seq_lens_cpu,  # CPU上的序列长度
                num_computed_tokens_cpu=num_computed_tokens_cpu,  # CPU上已计算的token数
                num_reqs=num_reqs,  # 请求数量
                num_actual_tokens=total_num_scheduled_tokens,  # 实际token数量
                max_query_len=max_num_scheduled_tokens,  # 最大查询长度
                max_seq_len=max_seq_len,  # 最大序列长度
                block_table_tensor=blk_table_tensor,  # 块表张量
                slot_mapping=slot_mapping,  # 槽映射
                logits_indices_padded=logits_indices_padded,  # 填充的logits索引
                num_logits_indices=logits_indices.size(0),  # logits索引数量
                causal=True,  # 因果注意力
                encoder_seq_lens=encoder_seq_lens,  # 编码器序列长度
                dcp_local_seq_lens=self.dcp_local_seq_lens.gpu[:num_reqs]  # DCP本地序列长度
                if self.dcp_world_size > 1
                else None,
            )

            # ==================== 设置推测解码通用注意力元数据 ====================
            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):  # 如果是Eagle提议器
                    if (
                        self.drafter.attn_layer_names[0]  # 如果第一个注意力层名
                        in kv_cache_group_spec.layer_names  # 在KV缓存组层名中
                    ):
                        spec_decode_common_attn_metadata = common_attn_metadata  # 设置推测解码元数据
                else:
                    spec_decode_common_attn_metadata = common_attn_metadata  # 设置推测解码元数据

            # ==================== 为每个注意力组构建元数据 ====================
            for attn_group in self.attn_groups[kv_cache_group_id]:  # 遍历注意力组
                # ==================== 准备级联注意力 ====================
                # 如果启用且有益，准备级联注意力
                common_prefix_len = 0  # 初始化公共前缀长度
                builder = attn_group.get_metadata_builder()  # 获取元数据构建器
                if self.cascade_attn_enabled:  # 如果启用级联注意力
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,  # 调度的token数量
                        num_common_prefix_blocks,  # 公共前缀块数
                        attn_group.kv_cache_spec,  # KV缓存规格
                        builder,  # 构建器
                    )

                # ==================== 准备额外注意力元数据参数 ====================
                extra_attn_metadata_args = {}  # 初始化额外参数
                if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):  # 如果使用推测解码且是GDN构建器
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs],  # 接受的token数量
                        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                            :num_reqs
                        ],  # CPU上的解码草稿token数量
                    )

                # ==================== 构建注意力元数据 ====================
                if ubatch_slices is not None:  # 如果有统一批次切片
                    common_attn_metadata_list = split_attn_metadata(
                        ubatch_slices, common_attn_metadata  # 分割注意力元数据
                    )
                    for ubid, common_attn_metadata in enumerate(
                        common_attn_metadata_list  # 遍历分割后的元数据
                    ):
                        attn_metadata_i = attn_group.get_metadata_builder(
                            ubatch_id=ubid  # 获取统一批次ID的构建器
                        ).build(
                            common_prefix_len=common_prefix_len,  # 公共前缀长度
                            common_attn_metadata=common_attn_metadata,  # 通用注意力元数据
                        )
                        for layer_name in kv_cache_group_spec.layer_names:  # 遍历层名
                            assert type(attn_metadata) is list  # 确保是列表类型
                            attn_metadata[ubid][layer_name] = attn_metadata_i  # 设置注意力元数据
                else:
                    assert isinstance(attn_metadata, dict)  # 确保是字典类型
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,  # 公共前缀长度
                        common_attn_metadata=common_attn_metadata,  # 通用注意力元数据
                        **extra_attn_metadata_args,  # 额外参数
                    )
                    use_cascade_attn |= getattr(attn_metadata_i, "use_cascade", False)  # 更新级联注意力标志
                    for layer_name in attn_group.layer_names:  # 遍历层名
                        attn_metadata[layer_name] = attn_metadata_i  # 设置注意力元数据

        # ==================== 禁用级联注意力 ====================
        # 当使用DBO时禁用级联注意力
        if ubatch_slices is not None:
            use_cascade_attn = False

        # ==================== 热交换LoRA模型 ====================
        if self.lora_config:  # 如果有LoRA配置
            self.set_active_loras(self.input_batch, num_scheduled_tokens)  # 设置活跃的LoRA

        # ==================== 返回结果 ====================
        return (
            attn_metadata,  # 注意力元数据
            logits_indices,  # logits索引
            spec_decode_metadata,  # 推测解码元数据
            num_scheduled_tokens,  # 调度的token数量
            spec_decode_common_attn_metadata,  # 推测解码通用注意力元数据
            max_num_scheduled_tokens,  # 最大调度的token数量
            ubatch_slices,  # 统一批次切片
            num_tokens_across_dp,  # 数据并行中的token数量
            use_cascade_attn,  # 是否使用级联注意力
        )

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """
        计算级联注意力的公共前缀长度
        
        注意：此函数返回的公共前缀长度表示专门用于级联注意力的长度，
        而不是请求之间实际共享的token数量。当级联注意力被禁用时（use_cascade=False），
        即使请求共享公共token，此函数也返回0。此外，公共前缀长度被截断为块大小的倍数，
        并可能由于下面解释的实现细节而进一步截断。

        Args:
            num_scheduled_tokens: 每个请求调度的token数量
            num_common_prefix_blocks: 共享的KV缓存块数量
            kv_cache_spec: KV缓存规格
            attn_metadata_builder: 注意力元数据构建器

        Returns:
            int: 公共前缀的token长度
        """
        # ==================== 计算基础公共前缀长度 ====================
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size  # 计算基础长度
        if common_prefix_len == 0:
            # 常见情况：没有公共前缀
            return 0

        # ==================== 级联注意力实现细节 ====================
        # 注意：级联注意力使用两个注意力内核：一个
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len, self.input_batch.num_computed_tokens_cpu[:num_reqs].min()
        )
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and num_logits <= self.cudagraph_batch_sizes[-1]
        ):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_logits_padded = self.vllm_config.pad_for_cudagraph(num_logits)
        else:
            num_logits_padded = num_logits
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_kwargs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[MultiModalKwargsItem], list[tuple[str, PlaceholderRange]]]:
        """Batch multimodal kwargs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_kwargs, req_ids_pos) where:
            - mm_kwargs: List of multimodal kwargs items to be batched
            - mm_hashes_pos: List of (mm_hash, position_info) tuples
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], []
        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # list of tuple (mm_hash, position_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        return mm_kwargs, mm_hashes_pos

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        """
        执行多模态编码器，处理多模态输入数据
        
        将多模态输入（如图像、视频、音频）编码为嵌入向量，
        用于后续的模型推理
        
        Args:
            scheduler_output: 调度器输出，包含多模态请求信息
        """
        # ==================== 批处理多模态输入 ====================
        # 使用辅助方法批处理多模态输入
        mm_kwargs, mm_hashes_pos = self._batch_mm_kwargs_from_scheduler(
            scheduler_output
        )

        # 如果没有多模态输入，直接返回
        if not mm_kwargs:
            return

        # ==================== 按模态分组处理 ====================
        # 尽可能批处理多模态输入：如果批次中的请求有多个模态
        # 或与之前的模态不同，我们分别处理以保持项目顺序
        # 注意：这是处理同一批次中多个模态的临时解决方案，
        # 正确的解决方案应该是重新排序编码器输出
        model = cast(SupportsMultiModal, self.model)  # 转换为支持多模态的模型
        encoder_outputs = []  # 存储编码器输出
        
        # 按模态分组处理多模态输入
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
        ):
            # ==================== 视频模态特殊处理 ====================
            # 临时解决方案：限制处理多模态数据时的峰值内存使用
            # 这解决了调度器将太多视频样本放入单个批次的问题
            # 调度器使用修剪后的视觉token计数来与计算预算比较，这是不正确的
            # （应该考虑输入媒体大小或非修剪的输出视觉token计数）
            curr_group_outputs = []

            if self.is_multimodal_pruning_enabled and modality == "video":
                # 视频模态使用微批次处理，避免内存溢出
                micro_batch_size = 1
                for i in range(0, num_items, micro_batch_size):
                    # 创建微批次输入
                    micro_batch_mm_inputs = dict(
                        (k, v[i : i + micro_batch_size])
                        for k, v in mm_kwargs_group.items()
                    )

                    # 获取多模态嵌入
                    micro_batch_outputs = model.get_multimodal_embeddings(
                        **micro_batch_mm_inputs
                    )

                    curr_group_outputs.extend(micro_batch_outputs)
            else:
                # ==================== 运行编码器 ====================
                # 对于非视频模态，直接批处理处理
                # `curr_group_outputs` 是以下之一：
                # 1. 形状为 (num_items, feature_size, hidden_size) 的张量
                #    在所有多模态项目的feature_size固定的情况下
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.
                curr_group_outputs = model.get_multimodal_embeddings(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        收集多模态嵌入，将编码器输出分配到正确的位置
        
        从编码器缓存中获取多模态嵌入，并根据位置信息
        将它们分配到输入序列中的正确位置
        
        Args:
            scheduler_output: 调度器输出，包含请求信息
            shift_computed_tokens: 已计算token的偏移量
            
        Returns:
            tuple: 包含以下元素的元组
                - mm_embeds: 多模态嵌入列表
                - is_mm_embed: 标识哪些位置是多模态嵌入的张量
        """
        # ==================== 初始化变量 ====================
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 总调度token数

        mm_embeds = list[torch.Tensor]()  # 多模态嵌入列表
        is_mm_embed = self.is_mm_embed.cpu  # 获取多模态嵌入标识（CPU）
        is_mm_embed[:total_num_scheduled_tokens] = False  # 初始化为False

        req_start_idx = 0  # 请求起始索引
        should_sync_mrope_positions = False  # 是否需要同步M-RoPE位置

        # ==================== 遍历每个请求 ====================
        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []  # 当前请求的多模态嵌入

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]  # 当前请求调度的token数
            req_state = self.requests[req_id]  # 获取请求状态
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens  # 已计算的token数

            # ==================== 处理多模态特征 ====================
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position  # 位置信息
                start_pos = pos_info.offset  # 起始位置
                num_encoder_tokens = pos_info.length  # 编码器token数量

                # ==================== 检查是否需要编码器输出 ====================
                # 如果两个范围重叠，则需要编码器输出：
                # [num_computed_tokens, num_computed_tokens + num_scheduled_tokens) 和
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # 此步骤不需要编码器输出
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # 编码器输出已经处理并存储在解码器的KV缓存中
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = self.is_mm_embed.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def _extract_encoder_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, torch.Tensor]:
        """Extract encoder inputs for encoder-decoder models.

        This method extracts multimodal input features from scheduled encoder
        inputs and formats them for the encoder-decoder model forward pass.
        """
        # Batch the multi-modal inputs using the helper method.
        mm_kwargs, _ = self._batch_mm_kwargs_from_scheduler(scheduler_output)

        if not mm_kwargs:
            return {}

        # Group MM kwargs by modality and extract features
        model = cast(SupportsMultiModal, self.model)
        encoder_features = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
        ):
            # Add the grouped features to encoder_features dict
            # This allows the model to receive them as kwargs (e.g.,
            # input_features=...)
            encoder_features.update(mm_kwargs_group)

        return encoder_features

    def get_model(self) -> nn.Module:
        """
        获取原始模型实例
        
        从CUDA Graph包装器或UBatch包装器中提取原始模型，
        用于直接访问模型的方法和属性
        
        Returns:
            nn.Module: 原始模型实例
        """
        # ==================== 检查模型包装器 ====================
        # 从cudagraph包装器中获取原始模型
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            return self.model.unwrap()  # 解包装，返回原始模型
        return self.model  # 如果没有包装，直接返回模型

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        """
        获取模型支持的生成任务列表
        
        检查模型支持哪些类型的生成任务，如文本生成、转录等
        
        Returns:
            list[GenerationTask]: 支持的任务列表
        """
        # ==================== 获取模型实例 ====================
        model = self.get_model()  # 获取原始模型
        supported_tasks = list[GenerationTask]()  # 初始化支持的任务列表

        # ==================== 检查文本生成支持 ====================
        if is_text_generation_model(model):
            supported_tasks.append("generate")  # 添加文本生成任务

        # ==================== 检查转录支持 ====================
        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]  # 如果只支持转录，直接返回

            supported_tasks.append("transcription")  # 添加转录任务

        return supported_tasks  # 返回支持的任务列表

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """
        获取模型支持的池化任务列表
        
        检查池化模型支持哪些类型的池化任务，如嵌入、编码、分类、评分等
        
        Returns:
            list[PoolingTask]: 支持的池化任务列表
        """
        # ==================== 获取模型实例 ====================
        model = self.get_model()  # 获取原始模型
        
        # ==================== 检查是否为池化模型 ====================
        if not is_pooling_model(model):
            return []  # 如果不是池化模型，返回空列表

        # ==================== 获取池化器支持的任务 ====================
        supported_tasks = list(model.pooler.get_supported_tasks())  # 获取池化器支持的任务

        # ==================== 处理分块预填充限制 ====================
        if (
            self.scheduler_config.chunked_prefill_enabled  # 启用了分块预填充
            and "encode" in supported_tasks  # 且支持编码任务
        ):
            supported_tasks.remove("encode")  # 移除编码任务

            logger.debug_once(
                "Chunked prefill is not supported with "
                "encode task which using ALL pooling. "
                "Please turn off chunked prefill by "
                "`--no-enable-chunked-prefill` before using it."
            )

        # ==================== 处理评分任务限制 ====================
        if "score" in supported_tasks:  # 如果支持评分任务
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)  # 获取标签数量
            if num_labels != 1:  # 如果标签数量不等于1
                supported_tasks.remove("score")  # 移除评分任务
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks  # 返回支持的任务列表

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        获取模型支持的所有任务列表
        
        根据模型的运行器类型（生成或池化）返回相应的支持任务
        
        Returns:
            tuple[SupportedTask, ...]: 支持的任务元组
        """
        # ==================== 初始化任务列表 ====================
        tasks = list[SupportedTask]()  # 创建空的任务列表

        # ==================== 根据运行器类型添加任务 ====================
        if self.model_config.runner_type == "generate":  # 如果是生成模型
            tasks.extend(self.get_supported_generation_tasks())  # 添加生成任务
        if self.model_config.runner_type == "pooling":  # 如果是池化模型
            tasks.extend(self.get_supported_pooling_tasks())  # 添加池化任务

        return tuple(tasks)  # 返回任务元组

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        执行EPLB（专家并行负载均衡）状态的一步
        
        EPLB是Mixture-of-Experts（MoE）模型中的负载均衡机制，
        用于在多个专家之间分配计算负载，确保每个专家的使用率相对均衡。
        
        Args:
            is_dummy: 是否为虚拟步骤（用于测试或预热）
            is_profile: 是否用于性能分析
        """
        # ==================== 检查EPLB是否启用 ====================
        if not self.parallel_config.enable_eplb:
            return  # 如果未启用EPLB，直接返回

        # ==================== 验证状态和模型 ====================
        assert self.eplb_state is not None  # 确保EPLB状态已初始化
        model = self.get_model()  # 获取当前模型
        assert is_mixture_of_experts(model)  # 确保模型是MoE模型
        
        # ==================== 执行EPLB步骤 ====================
        # 调用EPLB状态管理器执行一步负载均衡
        self.eplb_state.step(
            model,      # 模型实例
            is_dummy,   # 是否为虚拟步骤
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    # This is where the second ubatch is adjusted to account for the padding.
    # Should be called after attention metadata creation. This just pads
    # the second ubatch slice out to the total number of tokens
    # (num_tokens + padding)
    @staticmethod
    def pad_out_ubatch_slice(ubatch_slices: UBatchSlices, num_total_tokens: int):
        padded_second_ubatch_slice = slice(
            ubatch_slices[1].token_slice.start, num_total_tokens
        )
        ubatch_slices[1] = UBatchSlice(
            padded_second_ubatch_slice, padded_second_ubatch_slice
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        """
        执行池化操作，将隐藏状态转换为固定长度的向量表示
        
        用于嵌入生成、分类、评分等非生成任务
        
        Args:
            hidden_states: 模型的隐藏状态张量
            num_scheduled_tokens: 调度的token数量
            num_scheduled_tokens_np: 调度的token数量（numpy数组）
            
        Returns:
            ModelRunnerOutput: 包含池化输出的模型运行结果
        """
        # ==================== 验证批次一致性 ====================
        # 确保批次中的所有请求要么都是池化请求，要么都不是
        assert self.input_batch.num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        # ==================== 准备池化数据 ====================
        # 截取隐藏状态到调度的token数量
        hidden_states = hidden_states[:num_scheduled_tokens]
        
        # 获取池化元数据
        pooling_metadata = self.input_batch.get_pooling_metadata()
        
        # 构建池化游标，用于指定池化操作的位置
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np.tolist(), device=hidden_states.device
        )
        
        # 获取序列长度（CPU）
        seq_lens_cpu = self.seq_lens.cpu[: self.input_batch.num_reqs]

        # ==================== 执行池化操作 ====================
        # 将模型转换为池化模型类型
        model = cast(VllmModelForPooling, self.model)
        
        # 调用池化器进行池化操作
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states,        # 隐藏状态
            pooling_metadata=pooling_metadata,  # 池化元数据
        )
        
        # ==================== 处理输出 ====================
        # 将原始池化输出转换为CPU（非阻塞）
        raw_pooler_output = json_map_leaves(
            lambda x: x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        
        # 同步设备，确保所有操作完成
        self._sync_device()

        # ==================== 过滤有效输出 ====================
        # 只保留序列长度等于提示长度的输出（避免部分请求的输出）
        pooler_output: list[Optional[torch.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(
            raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens
        ):
            # 只有当序列长度等于提示长度时才保留输出
            # 这确保我们只处理完整的请求，而不是部分请求
            output = raw_output if seq_len == prompt_len else None
            pooler_output.append(output)

        # ==================== 构建返回结果 ====================
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,                    # 请求ID列表
            req_id_to_index=self.input_batch.req_id_to_index,    # 请求ID到索引映射
            sampled_token_ids=[],                                # 采样的token IDs（池化任务为空）
            logprobs=None,                                       # logprobs（池化任务为None）
            prompt_logprobs_dict={},                             # 提示logprobs字典（池化任务为空）
            pooler_output=pooler_output,                         # 池化输出（主要结果）
        )

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and not envs.VLLM_DISABLE_PAD_FOR_CUDAGRAPH
            and hasattr(self, "cudagraph_batch_sizes")
            and self.cudagraph_batch_sizes
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            # Use CUDA graphs.
            # Add padding to the batch size.
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (
            self.compilation_config.pass_config.enable_sequence_parallelism
            and tp_size > 1
        ):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[
        int,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[IntermediateTensors],
        dict[str, Any],
    ]:
        """
        预处理输入数据，准备模型前向传播所需的张量
        
        处理多模态输入、提示嵌入、位置编码等，为模型推理做准备
        
        Args:
            scheduler_output: 调度器输出，包含批次信息
            num_input_tokens: 输入token数量（已填充）
            intermediate_tensors: 流水线并行中的中间张量
            
        Returns:
            tuple: 包含以下元素的元组
                - num_scheduled_tokens: 调度的token数量
                - input_ids: 输入token IDs（如果使用）
                - inputs_embeds: 输入嵌入向量（如果使用）
                - positions: 位置编码张量
                - intermediate_tensors: 中间张量
                - model_kwargs: 模型关键字参数
        """
        # ==================== 基本参数获取 ====================
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取调度的token数量
        is_first_rank = get_pp_group().is_first_rank  # 检查是否为流水线并行的第一个rank

        # ==================== 多模态输入处理 ====================
        # _prepare_inputs可能会重新排序批次，所以我们必须在那之后收集多模态输出以确保正确的顺序
        if (
            self.supports_mm_inputs  # 支持多模态输入
            and is_first_rank  # 是第一个rank
            and not self.model_config.is_encoder_decoder  # 不是编码器-解码器模型
        ):
            # ==================== 执行多模态编码器 ====================
            # 运行多模态编码器（如果存在）
            self._execute_mm_encoder(scheduler_output)
            
            # 收集多模态嵌入
            mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # ==================== 统一输入格式 ====================
            # 注意：为了统一token IDs和软token（视觉嵌入），
            # 我们总是使用嵌入（而不是token IDs）作为多模态模型的输入，即使输入是文本
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                self.input_ids.gpu[:num_scheduled_tokens],  # 输入token IDs
                multimodal_embeddings=mm_embeds,            # 多模态嵌入
                is_multimodal=is_mm_embed,                  # 是否为多模态
            )

            # TODO: 避免复制操作，优化性能
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            # ==================== 设置多模态输入 ====================
            input_ids = None  # 多模态情况下不使用input_ids
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]  # 使用嵌入作为输入
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),  # 基础模型参数
                **self._extract_mm_kwargs(scheduler_output),      # 多模态参数
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if (
            self.model_config.is_encoder_decoder
            and scheduler_output.scheduled_encoder_inputs
        ):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )

    def _sample(
        self,
        logits: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
    ) -> SamplerOutput:
        """
        从logits中采样下一个token并获取logprobs（如果需要）
        
        Args:
            logits: 模型输出的logits张量
            spec_decode_metadata: 推测解码元数据（如果使用推测解码）
            
        Returns:
            SamplerOutput: 包含采样结果的对象
        """
        # ==================== 获取采样元数据 ====================
        sampling_metadata = self.input_batch.sampling_metadata  # 获取采样元数据
        
        # ==================== 普通采样（非推测解码） ====================
        if spec_decode_metadata is None:
            # 直接使用采样器进行普通采样
            return self.sampler(
                logits=logits,                    # 输入logits
                sampling_metadata=sampling_metadata,  # 采样元数据
            )

        # ==================== 推测解码采样 ====================
        # 当使用张量索引（bonus_logits_indices）时，PyTorch
        # 会创建一个与原始logits张量具有独立存储的新张量
        # 这意味着对bonus_logits的任何就地操作都不会影响原始logits张量
        assert logits is not None
        
        # 提取奖励logits（用于推测解码的额外token）
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        
        # 使用奖励logits进行采样
        sampler_output = self.sampler(
            logits=bonus_logits,                 # 奖励logits
            sampling_metadata=sampling_metadata,  # 采样元数据
            predict_bonus_token=True,            # 预测奖励token
        )
        bonus_token_ids = sampler_output.sampled_token_ids  # 获取奖励token IDs

        # ==================== 拒绝采样 ====================
        # 就像`bonus_logits`一样，`target_logits`是一个与原始`logits`张量
        # 具有独立存储的新张量。因此，安全地就地更新`target_logits`
        target_logits = logits[spec_decode_metadata.target_logits_indices]  # 提取目标logits
        
        # 使用拒绝采样器处理推测解码
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata,    # 推测解码元数据
            None,                    # draft_probs（草稿概率）
            target_logits,           # 目标logits
            bonus_token_ids,         # 奖励token IDs
            sampling_metadata,       # 采样元数据
        )
        
        # ==================== 更新输出和状态 ====================
        sampler_output.sampled_token_ids = output_token_ids  # 更新采样输出
        
        # 更新模型执行后的状态
        self._update_states_after_model_execute(output_token_ids)
        
        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        Optional[LogprobsLists],
        list[list[int]],
        dict[str, Optional[LogprobsTensors]],
        list[str],
        dict[str, int],
        list[int],
    ]:
        """
        执行同步簿记操作，处理采样结果和状态更新
        
        包括计算logprobs、处理丢弃的请求、更新生成器状态等
        
        Args:
            scheduler_output: 调度器输出
            sampler_output: 采样器输出
            logits: 模型输出的logits张量
            hidden_states: 隐藏状态张量
            num_scheduled_tokens: 调度的token数量
            
        Returns:
            tuple: 包含以下元素的元组
                - num_nans_in_logits: logits中的NaN数量统计
                - logprobs_lists: logprobs列表
                - valid_sampled_token_ids: 有效的采样token IDs
                - prompt_logprobs_dict: 提示logprobs字典
                - req_ids_output_copy: 请求ID输出副本
                - req_id_to_index_output_copy: 请求ID到索引映射副本
                - invalid_req_indices: 无效请求索引
        """
        # ==================== 检查logits中的NaN ====================
        num_nans_in_logits = {}  # 初始化NaN统计字典
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:  # 如果启用了NaN计算
            num_nans_in_logits = self._get_nans_in_logits(logits)  # 计算logits中的NaN数量

        # ==================== 处理丢弃的请求 ====================
        # 获取需要丢弃的请求索引
        discard_sampled_tokens_req_indices = self.discard_request_indices.np[
            : self.num_discarded_requests
        ]
        
        # 遍历丢弃的请求，更新其生成器状态
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))  # 获取请求的生成器
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = (
            logprobs_tensors.tolists() if logprobs_tensors is not None else None
        )

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)
            assert sampled_token_ids.shape[-1] == 1

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """
        调用模型前向传播的辅助方法
        
        此方法可以被子类重写以实现自定义的模型执行逻辑。
        动机：我们可以只检查这个方法，而不是整个execute_model方法，
        后者包含额外的逻辑。

        Args:
            input_ids: 输入token IDs（用于文本生成）
            positions: token位置编码
            intermediate_tensors: 来自前一个流水线阶段的张量（流水线并行）
            inputs_embeds: 输入嵌入向量（input_ids的替代方案）
            **model_kwargs: 其他模型参数

        Returns:
            Any: 模型输出张量（通常是隐藏状态）
        """
        # ==================== 执行模型前向传播 ====================
        # 直接调用模型的forward方法，传入所有必要的参数
        return self.model(
            input_ids=input_ids,                    # 输入token IDs
            positions=positions,                    # 位置编码
            intermediate_tensors=intermediate_tensors,  # 中间张量
            inputs_embeds=inputs_embeds,            # 输入嵌入
            **model_kwargs,                         # 其他模型参数
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        """
        执行模型推理的主要方法
        
        Args:
            scheduler_output: 调度器输出的批次信息
            intermediate_tensors: 流水线并行中的中间张量
            
        Returns:
            模型运行结果，包括生成的token、logprobs等
        """
        
        # ==================== 预处理阶段 ====================
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # 更新持久化批次状态，包括KV缓存、注意力元数据等
                self._update_states(scheduler_output)

                # 检查是否有需要调度的token
                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # 如果没有工作要做，返回空的模型运行结果
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    # 处理KV传输连接器，无需前向传播
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
                
                # 检查KV共享快速预填充配置
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                # 准备解码器输入，包括注意力元数据、logits索引等
                (
                    attn_metadata,                    # 注意力机制元数据
                    logits_indices,                   # logits的索引
                    spec_decode_metadata,             # 推测解码元数据
                    num_scheduled_tokens_np,          # 调度的token数量（numpy数组）
                    spec_decode_common_attn_metadata, # 推测解码通用注意力元数据
                    max_query_len,                    # 最大查询长度
                    ubatch_slices,                    # 统一批次切片
                    num_tokens_across_dp,             # 数据并行中的token数量
                    use_cascade_attn,                 # 是否使用级联注意力
                ) = self._prepare_inputs(scheduler_output)

            # 处理统一批次切片，确保输入token数量正确
            if ubatch_slices:
                assert num_tokens_across_dp is not None
                num_input_tokens = int(num_tokens_across_dp[0].item())
                self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
            elif num_tokens_across_dp is not None:
                num_input_tokens = int(num_tokens_across_dp[0].item())
            else:
                num_input_tokens = self._get_num_input_tokens(
                    scheduler_output.total_num_scheduled_tokens
                )

            # 预处理输入数据，准备模型输入
            (
                num_scheduled_tokens,     # 调度的token数量
                input_ids,               # 输入token IDs
                inputs_embeds,           # 输入嵌入向量
                positions,               # 位置编码
                intermediate_tensors,    # 中间张量
                model_kwargs,            # 模型关键字参数
            ) = self._preprocess(
                scheduler_output, num_input_tokens, intermediate_tensors
            )

            # 检查是否为统一解码（所有请求的查询长度相同）
            uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
                num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
            )
            
            # 创建批次描述符，用于CUDA Graph优化
            batch_descriptor = BatchDescriptor(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode
            )
            
            # 调度CUDA Graph运行时模式
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(batch_descriptor, use_cascade_attn)
            )

        # 如果启用了KV缩放计算，禁用CUDA Graph模式
        if attn_metadata is not None:
            metadata_list = (
                attn_metadata.values()
                if isinstance(attn_metadata, dict)
                else [attn_metadata]
            )
            if any(
                getattr(m, "enable_kv_scales_calculation", False) for m in metadata_list
            ):
                cudagraph_runtime_mode = CUDAGraphMode.NONE

        # ==================== 模型前向传播阶段 ====================
        # 运行模型，使用持久化缓冲区进行CUDA Graph优化
        with (
            set_forward_context(
                attn_metadata,                    # 注意力元数据
                self.vllm_config,                 # vLLM配置
                num_tokens=num_input_tokens,      # token数量
                num_tokens_across_dp=num_tokens_across_dp,  # 数据并行token数量
                cudagraph_runtime_mode=cudagraph_runtime_mode,  # CUDA Graph模式
                batch_descriptor=batch_descriptor,  # 批次描述符
                ubatch_slices=ubatch_slices,      # 统一批次切片
            ),
            record_function_or_nullcontext("Forward"),  # 性能记录
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,  # KV连接器输出
        ):
            # 执行模型前向传播
            model_output = self._model_forward(
                input_ids=input_ids,              # 输入token IDs
                positions=positions,              # 位置编码
                intermediate_tensors=intermediate_tensors,  # 中间张量
                inputs_embeds=inputs_embeds,      # 输入嵌入
                **model_kwargs,                   # 其他模型参数
            )

        # ==================== 后处理阶段 ====================
        with record_function_or_nullcontext("Postprocess"):
            # 处理模型输出，区分EAGLE 3和普通情况
            if self.use_aux_hidden_state_outputs:
                # EAGLE 3使用时的特殊情况，返回辅助隐藏状态
                hidden_states, aux_hidden_states = model_output
            else:
                # 普通情况，只返回隐藏状态
                hidden_states = model_output
                aux_hidden_states = None

            # 处理流水线并行输出广播
            if not self.broadcast_pp_output:
                # 普通情况
                if not get_pp_group().is_last_rank:
                    # 如果不是流水线并行的最后一个rank，返回中间张量
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                # 检查是否为池化模型
                if self.is_pooling_model:
                    # 执行池化操作，返回池化输出（用于嵌入、分类等任务）
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                # 选择需要计算logits的隐藏状态
                sample_hidden_states = hidden_states[logits_indices]
                # 计算logits（用于文本生成）
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # 罕见情况：需要广播流水线并行输出
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    # 如果不是最后一个rank，发送张量到下一个rank
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    # 如果是最后一个rank，计算logits
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                # 广播模型输出数据
                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # 应用结构化输出位掩码（如果存在）
            if scheduler_output.grammar_bitmask is not None:
                apply_grammar_bitmask(
                    scheduler_output, self.input_batch, logits, self.device
                )

        # ==================== 采样阶段 ====================
        with record_function_or_nullcontext("Sample"):
            # 从logits中采样下一个token
            sampler_output = self._sample(logits, spec_decode_metadata)

        # 定义推测解码的草稿token生成函数
        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                # 生成推测解码的草稿token IDs
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        # ==================== 推测解码处理 ====================
        # 检查是否使用EAGLE推测解码的填充批次
        use_padded_batch_for_eagle = (
            self.speculative_config
            and self.speculative_config.use_eagle()
            and not self.speculative_config.disable_padded_drafter_batch
        )
        
        # 确定草稿模型的最大长度
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            self.speculative_config
            and self.speculative_config.draft_model_config is not None
            and self.speculative_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len
            )
        
        # 检查输入是否适合草稿模型
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.max_seq_len
            + self.speculative_config.num_speculative_tokens
            <= effective_drafter_max_model_len
        )
        
        # 如果使用EAGLE填充批次且输入适合草稿模型，立即生成草稿token
        if use_padded_batch_for_eagle and input_fits_in_drafter:
            # EAGLE推测解码可以使用GPU采样的token作为输入，无需等待簿记完成
            propose_draft_token_ids(sampler_output.sampled_token_ids)

        # ==================== 簿记阶段 ====================
        with record_function_or_nullcontext("Bookkeep"):
            # 执行同步簿记操作，处理采样结果
            (
                num_nans_in_logits,           # logits中的NaN数量
                logprobs_lists,               # logprobs列表
                valid_sampled_token_ids,      # 有效的采样token IDs
                prompt_logprobs_dict,         # 提示logprobs字典
                req_ids_output_copy,          # 请求ID输出副本
                req_id_to_index_output_copy,  # 请求ID到索引的映射副本
                invalid_req_indices,          # 无效请求索引
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                num_scheduled_tokens,
            )

        # 如果使用其他推测解码方法（如ngram），在簿记后生成草稿token
        if (
            self.speculative_config
            and not use_padded_batch_for_eagle
            and input_fits_in_drafter
        ):
            # ngram和其他推测解码方法使用CPU上的采样token，所以在簿记后运行
            propose_draft_token_ids(valid_sampled_token_ids)

        # ==================== 专家并行负载均衡 ====================
        with record_function_or_nullcontext("EPLB"):
            # 执行专家并行负载均衡步骤
            self.eplb_step()

        # ==================== 构建输出结果 ====================
        # 创建模型运行输出结果
        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,                    # 请求ID列表
            req_id_to_index=req_id_to_index_output_copy,    # 请求ID到索引映射
            sampled_token_ids=valid_sampled_token_ids,      # 采样的token IDs
            logprobs=logprobs_lists,                        # logprobs
            prompt_logprobs_dict=prompt_logprobs_dict,      # 提示logprobs字典
            pooler_output=[],                               # 池化输出（文本生成时为空）
            kv_connector_output=kv_connector_output,        # KV连接器输出
            num_nans_in_logits=num_nans_in_logits,          # logits中的NaN数量
        )

        # 根据是否使用异步调度返回不同的输出类型
        if not self.use_async_scheduling:
            return output

        # 返回异步GPU模型运行输出
        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,                     # 模型运行输出
            sampled_token_ids=sampler_output.sampled_token_ids,  # 采样的token IDs
            invalid_req_indices=invalid_req_indices,        # 无效请求索引
            async_output_copy_stream=self.async_output_copy_stream,  # 异步输出复制流
        )

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        """
        获取草稿token IDs
        
        从内部缓存中获取推测解码生成的草稿token IDs，
        用于后续的推测解码处理
        
        Returns:
            Optional[DraftTokenIds]: 草稿token IDs对象，如果没有则返回None
        """
        # ==================== 检查是否有草稿token ====================
        if self._draft_token_ids is None:  # 如果没有草稿token
            return None  # 返回None
        
        # ==================== 获取请求ID ====================
        req_ids = self.input_batch.req_ids  # 获取请求ID列表
        
        # ==================== 转换草稿token格式 ====================
        if isinstance(self._draft_token_ids, torch.Tensor):  # 如果是张量格式
            draft_token_ids = self._draft_token_ids.tolist()  # 转换为列表
        else:
            draft_token_ids = self._draft_token_ids  # 直接使用
        
        # ==================== 清空缓存并返回 ====================
        self._draft_token_ids = None  # 清空内部缓存
        return DraftTokenIds(req_ids, draft_token_ids)  # 返回草稿token IDs对象

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: Union[torch.Tensor, list[list[int]]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[list[torch.Tensor]],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> Union[list[list[int]], torch.Tensor]:
        """
        提议草稿token IDs
        
        使用不同的推测解码方法（ngram、medusa、eagle等）生成草稿token IDs，
        用于加速文本生成过程
        
        Args:
            scheduler_output: 调度器输出
            sampled_token_ids: 采样的token IDs
            sampling_metadata: 采样元数据
            hidden_states: 隐藏状态
            sample_hidden_states: 采样隐藏状态
            aux_hidden_states: 辅助隐藏状态
            spec_decode_metadata: 推测解码元数据
            common_attn_metadata: 通用注意力元数据
            
        Returns:
            Union[list[list[int]], torch.Tensor]: 草稿token IDs
        """
        # ==================== 获取调度token数量 ====================
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens  # 获取调度的token数量
        
        # ==================== N-gram推测解码 ====================
        if self.speculative_config.method == "ngram":  # 如果使用N-gram方法
            assert isinstance(sampled_token_ids, list)  # 确保是列表格式
            assert isinstance(self.drafter, NgramProposer)  # 确保是N-gram提议器
            
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,  # 采样的token IDs
                self.input_batch.req_ids,  # 请求ID列表
                self.input_batch.num_tokens_no_spec,  # 非推测token数量
                self.input_batch.token_ids_cpu,  # CPU上的token IDs
                self.input_batch.spec_decode_unsupported_reqs,  # 不支持的推测解码请求
            )
        elif self.speculative_config.method == "medusa":  # 如果使用Medusa方法
            assert isinstance(sampled_token_ids, list)  # 确保是列表格式
            assert isinstance(self.drafter, MedusaProposer)  # 确保是Medusa提议器

            # ==================== 处理隐藏状态 ====================
            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # 目标模型的输入不包含草稿token
                hidden_states = sample_hidden_states  # 直接使用采样隐藏状态
            else:
                # ==================== 计算索引 ====================
                indices = []  # 初始化索引列表
                offset = 0  # 初始化偏移量
                assert spec_decode_metadata is not None  # 确保推测解码元数据不为空
                
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids  # 遍历草稿token数量和采样的token
                ):
                    indices.append(offset + len(tokens) - 1)  # 添加索引
                    offset += num_draft + 1  # 更新偏移量
                
                indices = torch.tensor(indices, device=self.device)  # 转换为张量
                hidden_states = sample_hidden_states[indices]  # 获取对应的隐藏状态

            # ==================== 使用Medusa提议器 ====================
            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,  # 目标隐藏状态
                sampling_metadata=sampling_metadata,  # 采样元数据
            )
        elif self.speculative_config.use_eagle():  # 如果使用EAGLE方法
            assert isinstance(self.drafter, EagleProposer)  # 确保是EAGLE提议器

            if self.speculative_config.disable_padded_drafter_batch:  # 如果禁用填充草稿批次
                # 当禁用填充批次时，采样的token IDs应该是
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                )

            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if self.speculative_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                else:
                    common_attn_metadata, token_indices, token_indices_to_sample = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count,
                        )
                    )

                target_token_ids = self.input_ids.gpu[token_indices]
                target_positions = self._get_positions(token_indices)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[token_indices]

            if self.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
            )

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        """
        更新配置
        
        动态更新模型运行器的配置参数，支持更新加载配置和模型配置
        
        Args:
            overrides: 配置覆盖字典，键为配置名称，值为新的配置值
        """
        # ==================== 定义允许的配置名称 ====================
        allowed_config_names = {"load_config", "model_config"}  # 允许的配置名称集合
        
        # ==================== 遍历并更新配置 ====================
        for config_name, config_overrides in overrides.items():  # 遍历配置覆盖
            # ==================== 验证配置名称 ====================
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "  # 配置名称不支持
                f"Allowed configs: {allowed_config_names}"  # 允许的配置列表
            )
            
            # ==================== 获取并更新配置 ====================
            config = getattr(self, config_name)  # 获取当前配置
            new_config = update_config(config, config_overrides)  # 更新配置
            setattr(self, config_name, new_config)  # 设置新配置

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info("Starting to load model %s...", self.model_config.model)
        if eep_scale_up:
            from vllm.distributed.parallel_state import get_ep_group

            num_local_physical_experts = torch.empty(1, dtype=torch.int32, device="cpu")
            torch.distributed.broadcast(
                num_local_physical_experts, group=get_ep_group().cpu_group, group_src=0
            )
            num_local_physical_experts = int(num_local_physical_experts.item())
            new_ep_size = get_ep_group().world_size
            global_expert_load, old_global_expert_indices = EplbState.recv_state()
            num_logical_experts = global_expert_load.shape[1]
            self.parallel_config.eplb_config.num_redundant_experts = (
                num_local_physical_experts * new_ep_size - num_logical_experts
            )
            assert old_global_expert_indices.shape[1] % num_local_physical_experts == 0
            old_ep_size = (
                old_global_expert_indices.shape[1] // num_local_physical_experts
            )
            rank_mapping = {
                old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)
            }
        else:
            global_expert_load = None
            old_global_expert_indices = None
            rank_mapping = None

        # ==================== 加载模型 ====================
        with DeviceMemoryProfiler() as m:  # 使用设备内存分析器监控内存使用
            time_before_load = time.perf_counter()  # 记录加载开始时间
            
            # 获取模型加载器
            model_loader = get_model_loader(self.load_config)
            logger.info("Loading model from scratch...")
            
            # 加载主模型
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
            
            # ==================== 加载LoRA模型 ====================
            if self.lora_config:
                # 如果配置了LoRA，加载LoRA模型
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )
            
            # ==================== 加载推测解码模型 ====================
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            
            # ==================== 检查EAGLE3支持 ====================
            if self.use_aux_hidden_state_outputs:
                if not supports_eagle3(self.model):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )

                # 尝试从推测配置获取辅助层，否则使用模型的默认层
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if aux_layers:
                    logger.info(
                        "Using auxiliary layers from speculative config: %s",
                        aux_layers,
                    )
                else:
                    aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                self.model.set_aux_hidden_state_layers(aux_layers)
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
        )
        prepare_communication_buffer_for_model(self.model)

        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.model)
            and self.model_config.multimodal_config.is_multimodal_pruning_enabled()
        )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info("EPLB is enabled for model %s.", self.model_config.model)
            self.eplb_state = EplbState.build(
                self.model,
                self.device,
                self.parallel_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )

        if (
            self.vllm_config.compilation_config.level == CompilationLevel.DYNAMO_AS_IS
            and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.dynamo_as_is_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation levels, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        if (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.enable_dbo
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.enable_dbo:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

    def _get_eagle3_aux_layers_from_config(self) -> Optional[tuple[int, ...]]:
        """
        从推测配置中提取EAGLE3辅助层索引

        这些索引指定在推测解码期间，基础模型的哪些隐藏状态
        应该用作EAGLE3草稿模型辅助输入

        Returns:
            Optional[tuple[int, ...]]: 如果在草稿模型配置中找到则返回层索引元组，否则返回None
        """
        # ==================== 检查推测配置 ====================
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None  # 如果没有推测配置或草稿模型配置，返回None

        # ==================== 获取HuggingFace配置 ====================
        hf_config = self.speculative_config.draft_model_config.hf_config  # 获取HuggingFace配置
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):  # 如果没有EAGLE辅助层ID属性
            return None  # 返回None

        # ==================== 提取层ID ====================
        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids  # 获取层ID
        if layer_ids and isinstance(layer_ids, (list, tuple)):  # 如果层ID存在且是列表或元组
            return tuple(layer_ids)  # 返回层ID元组

        return None  # 返回None

    def reload_weights(self) -> None:
        """
        重新加载模型权重
        
        在模型已经加载的情况下，重新加载模型权重。
        用于动态更新模型参数而不需要重新初始化整个模型
        
        Raises:
            AssertionError: 如果模型未加载则抛出异常
        """
        # ==================== 检查模型是否已加载 ====================
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."  # 模型未加载时不能重新加载权重
        )
        
        # ==================== 获取模型加载器 ====================
        model_loader = get_model_loader(self.load_config)  # 获取模型加载器
        
        # ==================== 重新加载权重 ====================
        logger.info("Reloading weights inplace...")  # 记录日志
        model_loader.load_weights(self.get_model(), model_config=self.model_config)  # 重新加载权重

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        """
        保存张量化模型
        
        将模型保存为张量化格式，用于模型序列化和存储
        
        Args:
            tensorizer_config: 张量化配置，指定保存格式和参数
        """
        # ==================== 保存张量化模型 ====================
        TensorizerLoader.save_model(
            self.get_model(),  # 获取模型实例
            tensorizer_config=tensorizer_config,  # 张量化配置
            model_config=self.model_config,  # 模型配置
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        """
        获取提示logprobs字典
        
        计算并返回每个请求的提示token的logprobs，
        用于分析提示部分的概率分布
        
        Args:
            hidden_states: 隐藏状态张量
            num_scheduled_tokens: 每个请求调度的token数量字典
            
        Returns:
            dict[str, Optional[LogprobsTensors]]: 请求ID到logprobs张量的映射
        """
        # ==================== 获取提示logprobs配置 ====================
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs  # 获取提示logprobs数量字典
        if not num_prompt_logprobs_dict:  # 如果没有提示logprobs配置
            return {}  # 返回空字典

        # ==================== 初始化变量 ====================
        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu  # 获取进行中的提示logprobs
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}  # 初始化结果字典

        # ==================== 处理每个请求 ====================
        # 由于提示logprobs是罕见功能，优先考虑简单、可维护的循环而不是最优性能
        completed_prefill_reqs = []  # 初始化完成的预填充请求列表
        
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():  # 遍历每个请求
            num_tokens = num_scheduled_tokens[req_id]  # 获取调度的token数量

            # ==================== 获取请求元数据 ====================
            request = self.requests[req_id]  # 获取请求对象
            if request.prompt_token_ids is None:  # 如果提示token IDs为空
                # 提示logprobs与提示嵌入不兼容
                continue  # 跳过此请求

            # ==================== 准备提示token数据 ====================
            num_prompt_tokens = len(request.prompt_token_ids)  # 获取提示token数量
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True  # 转换为张量并传输到设备
            )

            # ==================== 设置目标LogprobsTensors对象 ====================
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        """
        获取logits中的NaN数量
        
        统计每个请求的logits中NaN值的数量，用于调试和监控
        
        Args:
            logits: 模型输出的logits张量，可能为None
            
        Returns:
            dict[str, int]: 请求ID到NaN数量的映射
        """
        try:
            # ==================== 处理空logits ====================
            if logits is None:  # 如果logits为空
                return {req_id: 0 for req_id in self.input_batch.req_ids}  # 返回所有请求的NaN数量为0

            # ==================== 计算NaN数量 ====================
            num_nans_in_logits = {}  # 初始化NaN数量字典
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()  # 计算每个索引的NaN数量
            
            # ==================== 为每个请求统计NaN ====================
            for req_id in self.input_batch.req_ids:  # 遍历每个请求
                req_index = self.input_batch.req_id_to_index[req_id]  # 获取请求索引
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])  # 获取该请求的NaN数量
                    if num_nans_for_index is not None and req_index < logits.shape[0]  # 如果索引有效
                    else 0  # 否则设为0
                )
            return num_nans_in_logits  # 返回NaN数量字典
        except IndexError:  # 捕获索引错误
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        可能随机化输入IDs
        
        如果设置了VLLM_RANDOMIZE_DP_DUMMY_INPUTS环境变量，则随机化input_ids。
        这有助于平衡专家选择：
         - 在profile_run期间
         - 在数据并行rank虚拟运行期间
        
        Args:
            input_ids: 输入token IDs张量
        """
        # ==================== 检查是否需要随机化 ====================
        dp_size = self.vllm_config.parallel_config.data_parallel_size  # 获取数据并行大小
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1  # 检查是否需要随机化
        
        if not randomize_inputs:  # 如果不需要随机化
            yield  # 直接返回
        else:
            # ==================== 随机化输入 ====================
            import functools  # 导入functools模块

            @functools.cache  # 缓存装饰器
            def rand_input_ids() -> torch.Tensor:
                """
                生成随机输入IDs
                
                Returns:
                    torch.Tensor: 随机生成的输入IDs张量
                """
                return torch.randint_like(
                    self.input_ids.gpu,  # 参考张量
                    low=0,  # 最小值
                    high=self.model_config.get_vocab_size(),  # 最大值（词汇表大小）
                    dtype=input_ids.dtype,  # 数据类型
                )

            # ==================== 应用随机化 ====================
            logger.debug_once("Randomizing dummy data for DP Rank")  # 记录调试日志
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)  # 复制随机输入
            yield  # 执行上下文
            input_ids.fill_(0)  # 恢复为0

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """
        获取多模态虚拟批次数据
        
        为多模态模型的性能分析和预编译生成虚拟数据，
        用于测试和优化多模态处理性能
        
        Args:
            modality: 模态类型（如图像、音频等）
            max_items_per_batch: 批次中最大项目数量
            
        Returns:
            BatchedTensorInputs: 批处理的多模态张量输入
        """
        # ==================== 检查多模态预算 ====================
        assert self.mm_budget is not None  # 确保多模态预算已初始化

        # ==================== 获取虚拟解码器数据 ====================
        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,  # 模型配置
            seq_len=self.max_model_len,  # 最大序列长度
            mm_counts={modality: 1},  # 多模态计数
            cache=self.mm_budget.cache,  # 多模态预算缓存
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data  # 获取多模态数据

        # ==================== 创建虚拟多模态项目 ====================
        # 生成导致模型最大GPU消耗的数据
        dummy_mm_item = dummy_mm_data[modality][0]  # 获取第一个虚拟多模态项目
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch  # 复制到指定数量

        # ==================== 分组多模态参数 ====================
        model = cast(SupportsMultiModal, self.model)  # 转换为多模态支持模型
        return next(
            mm_kwargs_group  # 返回多模态参数组
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                dummy_mm_items,  # 虚拟多模态项目
                device=self.device,  # 设备
                pin_memory=self.pin_memory,  # 固定内存
                merge_by_field_config=model.merge_by_field_config,  # 按字段合并配置
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        运行虚拟前向传播
        
        执行虚拟前向传播以预热模型、性能分析或捕获CUDA图

        Args:
            num_tokens: 运行虚拟前向传播的token数量
            cudagraph_runtime_mode: 用于控制行为的CUDA图运行时模式
                - 如果未设置，将基于self.cudagraph_dispatcher确定CUDA图模式
                - CUDAGraphMode.NONE: 无CUDA图，用于预热和性能分析
                - CUDAGraphMode.PIECEWISE: 分段CUDA图
                - CUDAGraphMode.FULL: 完整CUDA图，需要注意力元数据
            force_attention: 如果为True，总是创建注意力元数据，用于在NONE模式下预热注意力后端
            uniform_decode: 如果为True，批次是统一解码批次
            skip_eplb: 如果为True，跳过EPLB状态更新
            is_profile: 如果为True，这是性能分析运行
            create_mixed_batch: 如果为True，创建包含解码（1个token）和预填充（多个token）请求的混合批次
            remove_lora: 如果为False，虚拟LoRA在运行后不会被销毁
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 返回的隐藏状态和logits张量
        """
        # ==================== 验证CUDA图运行时模式 ====================
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()  # 确保CUDA图模式有效
        )

        # ==================== 解释统一解码批次 ====================
        # 如果cudagraph_mode.decode_mode() == FULL且cudagraph_mode.separate_routine()，
        # 这意味着我们为混合预填充-解码批次与统一解码批次使用不同的图和/或模式。
        # 统一解码批次意味着所有请求都有相同的查询长度，除了批次中可能存在的虚拟请求（较短）用于填充。
        # 统一解码批次可以是常见的纯解码（max_query_len == 1），
        # 或推测解码（max_query_len == 1 + num_spec_decode_tokens）。

        # ==================== 设置最大查询长度 ====================
        # 当设置max_query_len = 1时，我们切换到并捕获FA2的优化例程用于纯解码，
        # 即Flashdecode + GQA/MQA的优化
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # ==================== 设置调度的token数量 ====================
        # 基于num_tokens和max_num_seqs设置num_scheduled_tokens，
        # 用于LoRA的虚拟运行，使得num_reqs总共具有num_tokens
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens  # 确保token数量不超过最大批次token数
        max_num_reqs = self.scheduler_config.max_num_seqs  # 获取最大请求数
        
        # ==================== 创建混合批次 ====================
        if create_mixed_batch:  # 如果创建混合批次
            assert not uniform_decode  # 确保不是统一解码
            # 创建混合批次：
            # 前半部分解码token，后半部分一个预填充
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)  # 解码token数量
            num_prefill_tokens = num_tokens - num_decode_tokens  # 预填充token数量
            num_reqs = num_decode_tokens + 1  # 请求总数

            # 创建解码请求（每个1个token）后跟预填充请求
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]  # 调度token列表
            # 注意：覆盖max_query_len为预填充token
            max_query_len = num_prefill_tokens  # 设置最大查询长度为预填充token数
        elif uniform_decode:  # 如果是统一解码
            assert not create_mixed_batch  # 确保不是混合批次
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))  # 计算请求数
            num_scheduled_tokens_list = [max_query_len] * num_reqs  # 创建调度token列表
            if num_tokens % max_query_len != 0:  # 如果有余数
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len  # 将余数分配给最后一个请求
        else:  # 普通情况
            num_reqs = min(num_tokens, max_num_reqs)  # 计算请求数
            min_tokens_per_req = num_tokens // num_reqs  # 每个请求的最小token数
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs  # 创建调度token列表
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs  # 将余数分配给最后一个请求

        # ==================== 验证调度配置 ====================
        assert sum(num_scheduled_tokens_list) == num_tokens  # 确保总token数正确
        assert len(num_scheduled_tokens_list) == num_reqs  # 确保请求数正确
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)  # 转换为numpy数组
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())  # 计算总调度token数

        # ==================== 协调数据并行批次 ====================
        # 我们目前只在token数量超过某个阈值时才进行微批次处理
        ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
            num_scheduled_tokens,  # 调度的token数量
            total_num_scheduled_tokens,  # 总调度token数
            total_num_scheduled_tokens,  # 总调度token数
            self.vllm_config.parallel_config,  # 并行配置
            allow_microbatching,  # 允许微批次
            uniform_decode,  # 统一解码
        )
        # ==================== 计算填充后的token数量 ====================
        num_tokens_after_padding = num_tokens  # 初始化填充后的token数量
        if num_tokens_across_dp is not None:  # 如果数据并行中有token数量信息
            num_tokens_after_padding = int(num_tokens_across_dp[0])  # 使用数据并行中的token数量

        # ==================== 初始化注意力元数据 ====================
        attn_metadata: Optional[PerLayerAttnMetadata] = None  # 初始化注意力元数据

        # ==================== 创建注意力元数据 ====================
        # 如果force_attention为True，我们总是捕获注意力。否则，
        # 只有在cudagraph_runtime_mode=FULL时才发生
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:  # 如果需要创建注意力元数据
            attn_metadata = {}  # 初始化注意力元数据字典
            if ubatch_slices is not None:  # 如果有统一批次切片
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]  # 为每个切片创建字典

            # ==================== 设置序列长度 ====================
            if create_mixed_batch:  # 如果创建混合批次
                # 在混合批次模式中（用于FI预热），我们使用
                # 较短的序列长度来运行更快
                # TODO(luka) 更好的虚拟批次描述系统
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]  # 设置序列长度
            else:  # 普通情况
                seq_lens = max_query_len  # 使用最大查询长度
            
            # ==================== 更新序列长度张量 ====================
            self.seq_lens.np[:num_reqs] = seq_lens  # 设置序列长度
            self.seq_lens.np[num_reqs:] = 0  # 填充剩余为0
            self.seq_lens.copy_to_gpu()  # 复制到GPU

            # ==================== 设置查询起始位置 ====================
            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)  # 获取累积token数
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens  # 设置查询起始位置
            self.query_start_loc.copy_to_gpu()  # 复制到GPU

            # ==================== 为每个KV缓存组创建注意力元数据 ====================
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups  # 遍历KV缓存组
            ):
                # ==================== 创建通用注意力元数据 ====================
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],  # GPU上的查询起始位置
                    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],  # CPU上的查询起始位置
                    seq_lens=self.seq_lens.gpu[:num_reqs],  # GPU上的序列长度
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],  # CPU上的序列长度
                    num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
                        :num_reqs
                    ],  # CPU上已计算的token数
                    num_reqs=num_reqs,  # 请求数量
                    num_actual_tokens=num_tokens,  # 实际token数量
                    max_query_len=max_query_len,  # 最大查询长度
                    max_seq_len=self.max_model_len,  # 最大序列长度
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id
                    ].get_device_tensor(num_reqs),  # 块表张量
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id
                    ].slot_mapping.gpu[:num_tokens],  # 槽映射
                    causal=True,  # 因果注意力
                    dcp_local_seq_lens=self.dcp_local_seq_lens.gpu[:num_reqs]
                    if self.dcp_world_size > 1
                    else None,
                )
                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata
                        )
                        for ubid, common_attn_metadata in enumerate(
                            common_attn_metadata_list
                        ):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = attn_group.get_metadata_builder(
                                ubatch_id=ubid
                            ).build_for_cudagraph_capture(common_attn_metadata)
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        metadata_builder = attn_group.get_metadata_builder()
                        attn_metadata_i = metadata_builder.build_for_cudagraph_capture(
                            common_attn_metadata
                        )
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(
            self.lora_config, num_scheduled_tokens, remove_lora
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_after_padding <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            else:
                input_ids = self.input_ids.gpu[:num_tokens_after_padding]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_after_padding]
            else:
                positions = self.positions.gpu[:num_tokens_after_padding]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_after_padding, None, False
                )

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    BatchDescriptor(
                        num_tokens=num_tokens_after_padding,
                        uniform_decode=uniform_decode,
                    )
                )
                if not is_profile
                else (CUDAGraphMode.NONE, None)
            )
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert (
                    cudagraph_runtime_mode == CUDAGraphMode.NONE
                    or cudagraph_runtime_mode == _cg_mode
                ), (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
                )
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        运行虚拟采样器
        
        使用虚拟隐藏状态运行采样器，用于测试和预热采样功能
        
        Args:
            hidden_states: 隐藏状态张量
            
        Returns:
            torch.Tensor: 采样的token IDs
        """
        # ==================== 处理虚拟隐藏状态 ====================
        # 虚拟隐藏状态可能包含特殊值，如`inf`或`nan`。
        # 为了避免破坏采样器，我们在这里使用随机张量
        hidden_states = torch.rand_like(hidden_states)  # 生成随机隐藏状态

        # ==================== 计算logits ====================
        logits = self.model.compute_logits(hidden_states)  # 计算logits
        num_reqs = logits.size(0)  # 获取请求数量

        # ==================== 创建虚拟张量生成器 ====================
        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)  # 创建虚拟张量生成器

        # ==================== 创建虚拟采样元数据 ====================
        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),  # 温度参数
            all_greedy=False,  # 不是所有请求都使用贪婪采样
            all_random=False,  # 不是所有请求都使用随机采样
            top_p=dummy_tensors(0.9),  # top-p参数
            top_k=dummy_tensors(logits.size(1) - 1),  # top-k参数
            generators={},  # 随机数生成器
            max_num_logprobs=None,  # 最大logprobs数量
            no_penalties=True,  # 无惩罚
            prompt_token_ids=None,  # 提示token IDs
            frequency_penalties=dummy_tensors(0.1),  # 频率惩罚
            presence_penalties=dummy_tensors(0.1),  # 存在惩罚
            repetition_penalties=dummy_tensors(0.1),  # 重复惩罚
            output_token_ids=[[] for _ in range(num_reqs)],  # 输出token IDs
            spec_token_ids=[[] for _ in range(num_reqs)],  # 推测token IDs
            allowed_token_ids_mask=None,  # 允许的token IDs掩码
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            target_logits = torch.randn(
                num_tokens, logits.shape[-1], device=self.device, dtype=logits.dtype
            )
            # NOTE(woosuk): Here, we should use int32 because the sampler uses
            # int32 for bonus_token_ids. If the dtype mismatches, re-compilation
            # will occur at runtime.
            bonus_token_ids = torch.zeros(
                num_reqs, device=self.device, dtype=torch.int32
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                target_logits,
                bonus_token_ids,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        """
        运行虚拟池化器任务
        
        使用虚拟隐藏状态运行池化器任务，用于测试和预热池化功能
        
        Args:
            hidden_states: 隐藏状态张量
            task: 池化任务
            
        Returns:
            PoolerOutput: 池化器输出
        """
        # ==================== 计算请求配置 ====================
        num_tokens = hidden_states.shape[0]  # 获取token数量
        max_num_reqs = self.scheduler_config.max_num_seqs  # 获取最大请求数
        num_reqs = min(num_tokens, max_num_reqs)  # 计算实际请求数
        min_tokens_per_req = num_tokens // num_reqs  # 每个请求的最小token数
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs  # 创建调度token列表
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs  # 将余数分配给最后一个请求
        
        # ==================== 验证配置 ====================
        assert sum(num_scheduled_tokens_list) == num_tokens  # 确保总token数正确
        assert len(num_scheduled_tokens_list) == num_reqs  # 确保请求数正确

        # ==================== 计算每个请求的token数 ====================
        req_num_tokens = num_tokens // num_reqs  # 每个请求的token数

        dummy_prompt_lens = torch.tensor(
            num_scheduled_tokens_list,
            device="cpu",
        )
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(task=task, model_config=self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_list, device=hidden_states.device
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        """
        运行虚拟池化器
        
        使用虚拟隐藏状态运行池化器，用于测试和预热池化功能。
        会测试所有支持的池化任务，选择输出最大的任务进行后续步骤
        
        Args:
            hidden_states: 隐藏状态张量
            
        Returns:
            PoolerOutput: 池化器输出
        """
        # ==================== 查找输出最大的任务 ====================
        # 查找在后续步骤中具有最大输出的任务
        output_size = dict[PoolingTask, float]()  # 初始化输出大小字典
        
        # ==================== 测试所有支持的池化任务 ====================
        for task in self.get_supported_pooling_tasks():  # 遍历所有支持的池化任务
            # 使用每个任务运行完整批次以确保没有内存不足
            output = self._dummy_pooler_run_task(hidden_states, task)  # 运行虚拟池化任务
            output_size[task] = sum(o.nbytes for o in output)  # 计算输出大小
            del output  # 允许垃圾回收

        # ==================== 选择输出最大的任务 ====================
        max_task = max(output_size.items(), key=lambda x: x[1])[0]  # 找到输出最大的任务
        return self._dummy_pooler_run_task(hidden_states, max_task)  # 使用最大任务运行

    def profile_run(self) -> None:
        """
        运行性能分析
        
        执行模型性能分析，包括多模态编码器和编码器缓存的配置，
        以及虚拟前向传播来预热模型和分配通信缓冲区
        """
        # ==================== 多模态编码器和编码器缓存分析 ====================
        # 使用多模态编码器和编码器缓存进行分析
        if self.supports_mm_inputs:  # 如果支持多模态输入
            if self.model_config.multimodal_config.skip_mm_profiling:  # 如果跳过多模态分析
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."  # 跳过多模态编码器和编码器缓存的内存分析
                )
            else:
                # ==================== 获取多模态预算 ====================
                mm_budget = self.mm_budget  # 获取多模态预算
                assert mm_budget is not None  # 确保多模态预算不为空

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:  # 如果编码器预算大于0
                    # ==================== 选择虚拟模态 ====================
                    # 注意：目前模型使用单个非文本模态进行分析，
                    # 即使它支持多个模态，也使用最大可能的输入token
                    dummy_modality = mm_budget.get_modality_with_max_tokens()  # 获取最大token的模态
                    max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[
                        dummy_modality
                    ]  # 获取该模态的最大批次项目数

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",  # 编码器缓存将使用%s token的预算初始化，
                        # 并使用%s个%s项目的最大特征大小进行分析
                        encoder_budget,  # 编码器预算
                        max_mm_items_per_batch,  # 最大批次项目数
                        dummy_modality,  # 虚拟模态
                    )

                    # ==================== 创建多模态虚拟批次 ====================
                    # 创建多模态输入的虚拟批次
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,  # 虚拟模态
                        max_mm_items_per_batch,  # 最大批次项目数
                    )

                    # ==================== 运行多模态编码器 ====================
                    # 运行多模态编码器
                    dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                        **batched_dummy_mm_inputs  # 使用虚拟多模态输入
                    )

                    # ==================== 验证编码器输出 ====================
                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,  # 虚拟编码器输出
                        expected_num_items=max_mm_items_per_batch,  # 期望的项目数
                    )

                    # ==================== 扩展编码器输出 ====================
                    # 注意：当编码器缓存需要存储编码器输出分散到的嵌入时会发生这种情况。
                    # 在这种情况下，我们创建大小为(encode_budget, hidden_size)的虚拟嵌入，
                    # 并将编码器输出分散到其中
                    encoder_output_shape = dummy_encoder_outputs[0].shape  # 获取编码器输出形状
                    if encoder_output_shape[0] < encoder_budget:  # 如果输出形状小于编码器预算
                        expanded_outputs = []  # 初始化扩展输出列表
                        for output in dummy_encoder_outputs:  # 遍历编码器输出
                            expanded = output.new_zeros(
                                (encoder_budget, encoder_output_shape[-1])  # 创建零张量
                            )
                            num_tokens = output.shape[0]  # 获取token数量
                            expanded[:num_tokens].copy_(output)  # 复制输出到扩展张量
                            expanded_outputs.append(expanded)  # 添加到扩展输出列表

                        dummy_encoder_outputs = expanded_outputs  # 更新虚拟编码器输出

                    # ==================== 缓存虚拟编码器输出 ====================
                    # 缓存虚拟编码器输出
                    self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))  # 存储到编码器缓存

        # ==================== 运行虚拟前向传播 ====================
        # 在这里添加`is_profile`来预分配通信缓冲区
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True  # 使用最大token数进行分析
        )
        
        # ==================== 根据模型类型运行相应功能 ====================
        if get_pp_group().is_last_rank:  # 如果是最后一个rank
            if self.is_pooling_model:  # 如果是池化模型
                output = self._dummy_pooler_run(hidden_states)  # 运行虚拟池化器
            else:  # 否则
                output = self._dummy_sampler_run(last_hidden_states)  # 运行虚拟采样器
        else:  # 如果不是最后一个rank
            output = None  # 输出为None
        
        # ==================== 清理和同步 ====================
        self._sync_device()  # 同步设备
        del hidden_states, output  # 删除变量
        self.encoder_cache.clear()  # 清空编码器缓存
        gc.collect()  # 垃圾回收

    def capture_model(self) -> int:
        """
        捕获模型CUDA图
        
        捕获模型的CUDA图以优化推理性能，支持混合预填充-解码和统一解码批次
        
        Returns:
            int: CUDA图占用的GPU内存大小（字节）
        """
        # ==================== 检查CUDA图模式 ====================
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:  # 如果CUDA图模式为NONE
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"  # 跳过CUDA图捕获，要启用CUDA图捕获，
                # 确保`cudagraph_mode`没有手动设置为`NONE`
            )
            return 0  # 返回0
        else:  # 否则
            self.initialize_cudagraph_capture()  # 初始化CUDA图捕获

        # ==================== 更新编译计数器 ====================
        compilation_counter.num_gpu_runner_capture_triggers += 1  # 增加GPU运行器捕获触发器计数

        # ==================== 记录开始时间 ====================
        start_time = time.perf_counter()  # 记录开始时间

        # ==================== 定义垃圾回收冻结上下文管理器 ====================
        @contextmanager
        def freeze_gc():
            """
            冻结垃圾回收上下文管理器
            
            在CUDA图捕获期间优化垃圾回收。
            清理，然后冻结所有剩余对象，使其不被包含在未来的收集中
            """
            gc.collect()  # 清理垃圾
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC  # 检查是否应该冻结
            if should_freeze:  # 如果需要冻结
                gc.freeze()  # 冻结垃圾回收
            try:
                yield  # 执行上下文
            finally:  # 最终清理
                if should_freeze:  # 如果之前冻结了
                    gc.unfreeze()  # 解冻垃圾回收
                    gc.collect()  # 再次清理

        # ==================== 触发CUDA图捕获 ====================
        # 为特定形状触发CUDA图捕获。
        # 首先捕获大形状，这样较小的形状可以重用为大形状分配的内存池
        set_cudagraph_capturing_enabled(True)  # 启用CUDA图捕获
        with freeze_gc(), graph_capture(device=self.device):  # 使用冻结GC和图捕获上下文
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]  # 记录开始时的空闲GPU内存
            cudagraph_mode = self.compilation_config.cudagraph_mode  # 获取CUDA图模式
            assert cudagraph_mode is not None  # 确保CUDA图模式不为空
            
            # ==================== 捕获混合预填充-解码CUDA图 ====================
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:  # 如果混合模式不是NONE
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()  # 获取混合运行时模式

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))  # 获取编译案例（反向）
                self._capture_cudagraphs(
                    compilation_cases,  # 编译案例
                    cudagraph_runtime_mode=cudagraph_runtime_mode,  # CUDA图运行时模式
                    uniform_decode=False,  # 不是统一解码
                )

            # ==================== 捕获统一解码CUDA图 ====================
            # 如果我们还没有完整的混合预填充-解码CUDA图，则捕获统一解码批次的完整CUDA图
            if (
                cudagraph_mode.decode_mode() == CUDAGraphMode.FULL  # 如果解码模式是FULL
                and cudagraph_mode.separate_routine()  # 且分离例程
            ):
                max_num_tokens = (
                    self.scheduler_config.max_num_seqs * self.uniform_decode_query_len  # 计算最大token数
                )
                decode_cudagraph_batch_sizes = [
                    x
                    for x in self.cudagraph_batch_sizes  # 从CUDA图批次大小中筛选
                    if max_num_tokens >= x >= self.uniform_decode_query_len  # 满足条件的批次大小
                ]
                compilation_cases_decode = list(reversed(decode_cudagraph_batch_sizes))  # 反向列表
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,  # 解码编译案例
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,  # 完整CUDA图运行时模式
                    uniform_decode=True,  # 统一解码
                )

            # ==================== 同步和记录内存使用 ====================
            torch.cuda.synchronize()  # 同步CUDA
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]  # 记录结束时的空闲GPU内存

        # ==================== 禁用CUDA图捕获 ====================
        # 全局禁用CUDA图捕获，这样任何意外的CUDA图捕获都会被检测到并在此后引发错误。
        # 注意：我们不将其放入graph_capture上下文管理器中，因为将来我们可能会进行延迟捕获，
        # 仍然允许在此后捕获
        set_cudagraph_capturing_enabled(False)  # 禁用CUDA图捕获

        # ==================== 计算和记录结果 ====================
        end_time = time.perf_counter()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算经过时间
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory  # 计算CUDA图大小
        # 这通常需要5~20秒
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",  # 图捕获在%.0f秒内完成，占用%.2f GiB
            elapsed_time,  # 经过时间
            cuda_graph_size / (1 << 30),  # CUDA图大小（GiB）
        )
        return cuda_graph_size  # 返回CUDA图大小

    def _capture_cudagraphs(
        self,
        compilation_cases: list[int],
        cudagraph_runtime_mode: CUDAGraphMode,
        uniform_decode: bool,
    ):
        """
        捕获CUDA图
        
        为指定的编译案例捕获CUDA图，支持预热和微批次处理
        
        Args:
            compilation_cases: 编译案例列表（token数量）
            cudagraph_runtime_mode: CUDA图运行时模式
            uniform_decode: 是否为统一解码
        """
        # ==================== 验证CUDA图运行时模式 ====================
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"  # 确保CUDA图运行时模式有效

        # ==================== 设置进度条 ====================
        # 只有rank 0应该在捕获期间打印进度条
        if is_global_first_rank():  # 如果是全局第一个rank
            compilation_cases = tqdm(
                compilation_cases,  # 编译案例
                disable=not self.load_config.use_tqdm_on_load,  # 是否禁用进度条
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",  # 解码或混合预填充-解码
                    cudagraph_runtime_mode.name,  # CUDA图运行时模式名称
                ),
            )

        # ==================== 遍历编译案例 ====================
        # 我们在这里跳过EPLB，因为我们不想记录虚拟指标
        for num_tokens in compilation_cases:  # 遍历每个token数量
            # ==================== 检查微批次处理 ====================
            # 我们目前只在它是FULL CUDA图、统一解码批次且token数量超过阈值时才捕获微批次图。
            # 否则我们只捕获图的非微批次版本
            allow_microbatching = (
                self.parallel_config.enable_dbo  # 启用DBO
                and cudagraph_runtime_mode == CUDAGraphMode.FULL  # 且是FULL模式
                and uniform_decode  # 且是统一解码
                and check_ubatch_thresholds(  # 且满足微批次阈值
                    config=self.vllm_config.parallel_config,  # 并行配置
                    num_tokens=num_tokens,  # token数量
                    uniform_decode=uniform_decode,  # 是否统一解码
                )
            )

            # ==================== 预热阶段 ====================
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):  # 预热次数
                # 使用CUDAGraphRuntimeStyle.NONE（默认）进行预热。
                # 但要小心，使用`NONE`预热与是否要预热注意力是正交的。
                # 这与`FULL`意味着捕获注意力而`PIECEWISE`意味着不捕获注意力的情况不同
                force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL  # 是否强制注意力
                self._dummy_run(
                    num_tokens,  # token数量
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,  # 使用NONE模式预热
                    force_attention=force_attention,  # 强制注意力
                    uniform_decode=uniform_decode,  # 是否统一解码
                    allow_microbatching=allow_microbatching,  # 允许微批次
                    skip_eplb=True,  # 跳过EPLB
                    remove_lora=False,  # 不移除LoRA
                )
            
            # ==================== 实际捕获阶段 ====================
            self._dummy_run(
                num_tokens,  # token数量
                cudagraph_runtime_mode=cudagraph_runtime_mode,  # 使用指定的运行时模式
                uniform_decode=uniform_decode,  # 是否统一解码
                allow_microbatching=allow_microbatching,  # 允许微批次
                skip_eplb=True,  # 跳过EPLB
                remove_lora=False,  # 不移除LoRA
            )
        
        # ==================== 清理LoRA ====================
        self.maybe_remove_all_loras(self.lora_config)  # 可能移除所有LoRA

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        初始化注意力后端和注意力元数据构建器
        
        为每个KV缓存组创建相应的注意力后端和元数据构建器，
        支持快速预填充和动态注意力后端子类
        
        Args:
            kv_cache_config: KV缓存配置
        """
        # ==================== 检查初始化状态 ====================
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"  # 确保注意力后端未初始化

        # ==================== 定义注意力组键 ====================
        class AttentionGroupKey(NamedTuple):
            """
            注意力组键
            
            用于标识注意力后端和KV缓存规格的组合
            """
            attn_backend: type[AttentionBackend]  # 注意力后端类型
            kv_cache_spec: KVCacheSpec  # KV缓存规格

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> dict[AttentionGroupKey, list[str]]:
            """
            获取组的注意力后端
            
            为指定的KV缓存组获取所有注意力后端和对应的层名
            
            Args:
                kv_cache_group_spec: KV缓存组规格
                
            Returns:
                dict[AttentionGroupKey, list[str]]: 注意力后端到层名的映射
            """
            # ==================== 获取注意力层 ====================
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase, kv_cache_group_spec.layer_names  # 从vLLM配置获取注意力层
            )
            attn_backends = {}  # 初始化注意力后端字典
            attn_backend_layers = defaultdict(list)  # 初始化注意力后端层字典
            
            # ==================== 处理每个层 ====================
            # 基于完整类名去重；这比使用类本身作为键更安全，
            # 因为当我们创建动态注意力后端子类（例如ChunkedLocalAttention）时，
            # 除非它们被正确缓存，否则每层都会有不同的对象
            for layer_name in kv_cache_group_spec.layer_names:  # 遍历层名
                attn_backend = layers[layer_name].get_attn_backend()  # 获取注意力后端

                # ==================== 处理快速预填充 ====================
                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:  # 如果层支持快速预填充
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",  # 快速预填充名称
                        attn_backend,  # 原始注意力后端
                    )

                # ==================== 创建键和映射 ====================
                full_cls_name = attn_backend.full_cls_name()  # 获取完整类名
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec  # 获取层KV缓存规格
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):  # 如果是统一类型KV缓存规格
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]  # 获取特定层的规格
                key = (full_cls_name, layer_kv_cache_spec)  # 创建键
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec  # 创建注意力组键
                )
                attn_backend_layers[key].append(layer_name)  # 添加层名
            return {attn_backends[k]: v for k, v in attn_backend_layers.items()}  # 返回映射

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
        ) -> list[AttentionGroup]:
            """
            创建注意力组
            
            根据注意力后端映射创建注意力组
            
            Args:
                attn_backends_map: 注意力后端映射
                
            Returns:
                list[AttentionGroup]: 注意力组列表
            """
            attn_groups: list[AttentionGroup] = []  # 初始化注意力组列表
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():  # 遍历映射
                # ==================== 创建注意力组 ====================
                attn_group = AttentionGroup.create_with_metadata_builders(
                    attn_backend,  # 注意力后端
                    layer_names,  # 层名列表
                    kv_cache_spec,  # KV缓存规格
                    self.vllm_config,  # vLLM配置
                    self.device,  # 设备
                    num_metadata_builders=1  # 元数据构建器数量
                    if not self.parallel_config.enable_dbo  # 如果未启用DBO则为1
                    else 2,  # 否则为2
                )

                attn_groups.append(attn_group)  # 添加到组列表
            return attn_groups  # 返回注意力组列表

        # ==================== 为每个KV缓存组创建注意力组 ====================
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:  # 遍历KV缓存组
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)  # 获取注意力后端
            self.attn_groups.append(create_attn_groups(attn_backends))  # 创建并添加注意力组

        # ==================== 计算重排序批次阈值 ====================
        # 计算重排序批次阈值（如果需要）
        self.calculate_reorder_batch_threshold()  # 计算重排序批次阈值

    def initialize_cudagraph_capture(self) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        backends with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.get_metadata_builder()
            if builder.cudagraph_support.value < min_cg_support.value:
                min_cg_support = builder.cudagraph_support
                min_cg_builder_name = builder.__class__.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_builder_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation level is piecewise"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_builder_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.level == CompilationLevel.PIECEWISE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_builder_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_builder_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation level is piecewise"
            )

        # Trigger cudagraph dispatching keys initialization here (after
        # initializing attn backends).
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode, self.uniform_decode_query_len
        )

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()

            # check that if any backends reorder batches; that the reordering
            # is compatible (e.g., decode threshold is the same)
            reorder_batch_threshold_i = attn_metadata_builder_i.reorder_batch_threshold
            if reorder_batch_threshold_i is not None:
                if self.reorder_batch_threshold is not None:
                    if reorder_batch_threshold_i != self.reorder_batch_threshold:
                        raise ValueError(
                            f"Attention backend reorders decodes with "
                            f"threshold {reorder_batch_threshold_i} but other "
                            f"backend uses threshold "
                            f"{self.reorder_batch_threshold}"
                        )
                else:
                    self.reorder_batch_threshold = reorder_batch_threshold_i

    def _find_compatible_block_sizes(
        self,
        kv_manager_block_size: int,
        backend_cls: type[AttentionBackend],
        return_all: bool = False,
    ) -> list[int]:
        """
        Find compatible block sizes for a backend.

        Args:
            kv_manager_block_size: Physical block size of KV cache
            backend_cls: Attention backend class
            return_all: Return all compatible sizes if True, max size if False

        Returns:
            Compatible block size(s) based on return_all parameter

        Raises:
            ValueError: If no compatible block size found
        """
        supported_block_size = backend_cls.get_supported_kernel_block_size()
        compatible_sizes = []

        for block_size in supported_block_size:
            if isinstance(block_size, int):
                if kv_manager_block_size % block_size == 0:
                    compatible_sizes.append(block_size)
            elif (
                isinstance(block_size, MultipleOf)
                and kv_manager_block_size % block_size.base == 0
            ):
                compatible_sizes.append(kv_manager_block_size)

        if not compatible_sizes:
            raise ValueError(f"No compatible block size for {kv_manager_block_size}")

        return compatible_sizes if return_all else [max(compatible_sizes)]

    def _select_common_block_size(
        self, kv_manager_block_size: int, attn_groups: list[AttentionGroup]
    ) -> int:
        """
        为所有后端选择通用块大小
        
        从所有注意力后端支持的块大小中选择一个通用的块大小，
        优先使用cache_config.block_size

        Args:
            kv_manager_block_size: KV缓存的块大小
            attn_groups: 注意力组列表

        Returns:
            int: 所有后端都支持的块大小，优先使用cache_config.block_size

        Raises:
            ValueError: 如果找不到通用块大小
        """
        # ==================== 收集所有后端支持的块大小 ====================
        all_backend_supports = []  # 初始化所有后端支持列表

        for attn_group in attn_groups:  # 遍历每个注意力组
            # ==================== 获取兼容的块大小 ====================
            compatible_sizes = self._find_compatible_block_sizes(
                kv_manager_block_size, attn_group.backend, return_all=True  # 查找兼容的块大小
            )
            supported_sizes = sorted(list(set(compatible_sizes)), reverse=True)  # 排序并去重
            all_backend_supports.append(set(supported_sizes))  # 添加到支持列表

        # ==================== 计算通用支持的块大小 ====================
        common_supported_sizes = set.intersection(*all_backend_supports)  # 计算交集

        # ==================== 检查是否有通用块大小 ====================
        if not common_supported_sizes:  # 如果没有通用块大小
            error_msg = f"No common block size for {kv_manager_block_size}. "  # 构建错误消息
            for i, attn_group in enumerate(attn_groups):  # 遍历注意力组
                supported = all_backend_supports[i]  # 获取支持的大小
                error_msg += (
                    f"Backend {attn_group.backend} supports: {sorted(supported)}. "  # 添加后端支持信息
                )
            raise ValueError(error_msg)  # 抛出错误

        # ==================== 选择最佳块大小 ====================
        if self.cache_config.block_size in common_supported_sizes:  # 如果配置的块大小在通用支持中
            return self.cache_config.block_size  # 返回配置的块大小

        return max(common_supported_sizes)  # 返回最大的通用块大小

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """
        可能重新初始化输入批次
        
        如果块大小与`[self.cache_config.block_size]`不同，则重新初始化输入批次。
        这通常发生在有多个KV缓存组时

        Args:
            kv_cache_config: KV缓存配置
        """
        # ==================== 收集块大小 ====================
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size  # 获取KV缓存组的块大小
            for kv_cache_group in kv_cache_config.kv_cache_groups  # 遍历KV缓存组
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)  # 排除编码器专用注意力规格
        ]

        # ==================== 生成内核块大小 ====================
        # 生成与每个块大小匹配的内核块大小
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)  # 准备内核块大小

        # ==================== 检查是否需要重新初始化 ====================
        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:  # 如果块大小不匹配
            # ==================== 检查CPU卸载 ====================
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # 启用CPU权重卸载时无法重新初始化输入批次
                "for more details."
            )
            
            # ==================== 重新初始化输入批次 ====================
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,  # 最大请求数
                max_model_len=max(self.max_model_len, self.max_encoder_len),  # 最大模型长度
                max_num_batched_tokens=self.max_num_tokens,  # 最大批次token数
                device=self.device,  # 设备
                pin_memory=self.pin_memory,  # 固定内存
                vocab_size=self.model_config.get_vocab_size(),  # 词汇表大小
                block_sizes=block_sizes,  # 块大小列表
                kernel_block_sizes=kernel_block_sizes,  # 内核块大小列表
                is_spec_decode=bool(self.vllm_config.speculative_config),  # 是否推测解码
                logitsprocs=self.input_batch.logitsprocs,  # logits处理器
                is_pooling_model=self.is_pooling_model,  # 是否池化模型
                num_speculative_tokens=(  # 推测token数量
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config  # 如果有推测配置
                    else 0  # 否则为0
                ),
            )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        分配KV缓存张量
        
        使用正确的大小初始化KV缓存缓冲区。缓冲区需要在使用前
        重塑为所需的形状

        Args:
            kv_cache_config: KV缓存配置
            
        Returns:
            dict[str, torch.Tensor]: 层名到对应KV缓存内存缓冲区的映射
        """
        # ==================== 初始化KV缓存原始张量 ====================
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}  # 初始化KV缓存原始张量字典
        
        # ==================== 为每个KV缓存张量分配内存 ====================
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:  # 遍历KV缓存张量
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device  # 创建零张量
            )
            for layer_name in kv_cache_tensor.shared_by:  # 遍历共享此张量的层
                kv_cache_raw_tensors[layer_name] = tensor  # 将张量分配给层

        # ==================== 验证层初始化 ====================
        layer_names = set()  # 初始化层名集合
        for group in kv_cache_config.kv_cache_groups:  # 遍历KV缓存组
            for layer_name in group.layer_names:  # 遍历组中的层名
                if layer_name in self.runner_only_attn_layers:  # 如果是运行器专用注意力层
                    continue  # 跳过
                layer_names.add(layer_name)  # 添加到层名集合
        
        # ==================== 检查所有层都已正确初始化 ====================
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"  # 某些层未正确初始化
        )
        return kv_cache_raw_tensors  # 返回KV缓存原始张量

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        """
        注意力组迭代器
        
        返回所有注意力组的迭代器
        
        Returns:
            Iterator[AttentionGroup]: 注意力组迭代器
        """
        return itertools.chain.from_iterable(self.attn_groups)  # 展平注意力组列表

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        """
        KV缓存规格注意力组迭代器
        
        返回与KV缓存规格相关的注意力组迭代器
        
        Returns:
            Iterator[AttentionGroup]: KV缓存规格注意力组迭代器
        """
        if not self.kv_cache_config.kv_cache_groups:  # 如果没有KV缓存组
            return  # 返回空迭代器
        for attn_groups in self.attn_groups:  # 遍历注意力组
            yield from attn_groups  # 生成每个注意力组

    def _prepare_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> list[int]:
        """
        准备内核块大小
        
        生成与每个块大小匹配的内核块大小。
        
        对于支持虚拟块分割的注意力后端，使用后端支持的块大小。
        对于其他后端（如Mamba），使用相同的块大小（无分割）

        Args:
            kv_cache_config: KV缓存配置

        Returns:
            list[int]: 每个缓存组的内核块大小列表
        """
        # ==================== 初始化内核块大小列表 ====================
        kernel_block_sizes = []  # 初始化内核块大小列表
        
        # ==================== 遍历KV缓存组 ====================
        for kv_cache_group_id, kv_cache_group in enumerate(
            kv_cache_config.kv_cache_groups  # 遍历KV缓存组
        ):
            # ==================== 处理编码器专用注意力规格 ====================
            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):  # 如果是编码器专用注意力规格
                continue  # 跳过
            
            # ==================== 处理注意力规格 ====================
            elif isinstance(kv_cache_group.kv_cache_spec, AttentionSpec):  # 如果是注意力规格
                # 这是一个支持虚拟块分割的注意力后端。
                # 从组中的所有后端获取支持的块大小
                attn_groups = self.attn_groups[kv_cache_group_id]  # 获取注意力组
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size  # 获取KV管理器块大小
                selected_kernel_size = self._select_common_block_size(
                    kv_manager_block_size, attn_groups  # 选择通用块大小
                )
                kernel_block_sizes.append(selected_kernel_size)  # 添加到内核块大小列表
            
            # ==================== 处理Mamba规格 ====================
            elif isinstance(kv_cache_group.kv_cache_spec, MambaSpec):  # 如果是Mamba规格
                # 这可能是Mamba或其他非注意力缓存，无分割
                kernel_block_sizes.append(kv_cache_group.kv_cache_spec.block_size)  # 添加块大小
            
            # ==================== 处理未知规格 ====================
            else:
                raise NotImplementedError(
                    f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"  # 未知的KV缓存规格
                )
        return kernel_block_sizes  # 返回内核块大小列表

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        重塑KV缓存张量
        
        将KV缓存张量重塑为所需的形状和数据类型

        Args:
            kv_cache_config: KV缓存配置
            kv_cache_raw_tensors: 每层的KV缓存缓冲区，具有正确大小但未初始化的形状
            
        Returns:
            dict[str, torch.Tensor]: 层名到对应KV缓存内存缓冲区的映射
        """
        # ==================== 初始化变量 ====================
        kv_caches: dict[str, torch.Tensor] = {}  # 初始化KV缓存字典
        has_attn, has_mamba = False, False  # 初始化注意力 and Mamba标志
        
        # ==================== 遍历KV缓存规格注意力组 ====================
        for group in self._kv_cache_spec_attn_group_iterator():  # 遍历KV缓存规格注意力组
            kv_cache_spec = group.kv_cache_spec  # 获取KV缓存规格
            attn_backend = group.backend  # 获取注意力后端
            
            # ==================== 处理每个层 ====================
            for layer_name in group.layer_names:  # 遍历层名
                if layer_name in self.runner_only_attn_layers:  # 如果是运行器专用注意力层
                    continue  # 跳过
                
                # ==================== 获取原始张量 ====================
                raw_tensor = kv_cache_raw_tensors[layer_name]  # 获取原始张量
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0  # 确保张量大小是页大小的倍数
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes  # 计算块数量
                
                # ==================== 处理注意力规格 ====================
                if isinstance(kv_cache_spec, AttentionSpec):  # 如果是注意力规格
                    has_attn = True  # 设置注意力标志
                    kv_manager_block_size = kv_cache_spec.block_size  # 获取KV管理器块大小
                    kernel_size_list = self._find_compatible_block_sizes(
                        kv_manager_block_size, attn_backend, return_all=False  # 查找兼容的块大小
                    )
                    kernel_size = kernel_size_list[0]  # 获取内核大小
                    num_blocks_per_kv_block = kv_manager_block_size // kernel_size  # 计算每个KV块的块数
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block  # 计算内核块数

                    # ==================== 获取KV缓存形状 ====================
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,  # 内核块数
                        kernel_size,  # 内核大小
                        kv_cache_spec.num_kv_heads,  # KV头数
                        kv_cache_spec.head_size,  # 头大小
                        cache_dtype_str=self.cache_config.cache_dtype,  # 缓存数据类型
                    )
                    dtype = kv_cache_spec.dtype  # 获取数据类型
                    
                    # ==================== 获取步长顺序 ====================
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()  # 获取KV缓存步长顺序
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)  # 确保长度匹配
                    except (AttributeError, NotImplementedError):  # 如果方法不存在
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))  # 使用默认顺序
                    
                    # ==================== 重塑张量 ====================
                    # 分配遵循后端定义的步长顺序，以确保每个后端的语义保持一致。
                    # 我们首先获得通用的KV缓存形状，然后根据步长顺序对其进行排列，
                    # 这可能导致非连续张量
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order  # 按步长顺序排列形状
                    )
                    # 保持原始KV形状视图
                    inv_order = [
                        kv_cache_stride_order.index(i)  # 获取逆序
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)  # 转换为指定数据类型
                        .view(kv_cache_shape)  # 重塑为指定形状
                        .permute(*inv_order)  # 按逆序排列
                    )
                
                # ==================== 处理Mamba规格 ====================
                elif isinstance(kv_cache_spec, MambaSpec):  # 如果是Mamba规格
                    has_mamba = True  # 设置Mamba标志
                    raw_tensor = kv_cache_raw_tensors[layer_name]  # 获取原始张量
                    state_tensors = []  # 初始化状态张量列表
                    storage_offset_bytes = 0  # 初始化存储偏移字节
                    
                    # ==================== 处理每个形状和数据类型 ====================
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):  # 遍历形状和数据类型
                        dtype_size = get_dtype_size(dtype)  # 获取数据类型大小
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size  # 计算每页元素数
                        )
                        target_shape = (num_blocks, *shape)  # 目标形状
                        stride = torch.empty(target_shape).stride()  # 获取步长
                        target_stride = (num_element_per_page, *stride[1:])  # 目标步长
                        assert storage_offset_bytes % dtype_size == 0  # 确保偏移是数据类型大小的倍数
                        
                        # ==================== 创建步长张量 ====================
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),  # 转换为指定数据类型
                            size=target_shape,  # 目标形状
                            stride=target_stride,  # 目标步长
                            storage_offset=storage_offset_bytes // dtype_size,  # 存储偏移
                        )
                        state_tensors.append(tensor)  # 添加到状态张量列表
                        storage_offset_bytes += stride[0] * dtype_size  # 更新存储偏移

                    kv_caches[layer_name] = state_tensors  # 设置层缓存
                
                # ==================== 处理未知规格 ====================
                else:
                    raise NotImplementedError  # 抛出未实现错误

        # ==================== 更新混合注意力Mamba布局 ====================
        if has_attn and has_mamba:  # 如果同时有注意力和Mamba
            self._update_hybrid_attention_mamba_layout(kv_caches)  # 更新混合布局

        return kv_caches  # 返回KV缓存

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        更新混合注意力Mamba布局
        
        将注意力层的布局从(2, num_blocks, ...)更新为(num_blocks, 2, ...)

        Args:
            kv_caches: 每层的KV缓存缓冲区
        """
        # ==================== 遍历KV缓存规格注意力组 ====================
        for group in self._kv_cache_spec_attn_group_iterator():  # 遍历KV缓存规格注意力组
            kv_cache_spec = group.kv_cache_spec  # 获取KV缓存规格
            
            # ==================== 处理每个层 ====================
            for layer_name in group.layer_names:  # 遍历层名
                kv_cache = kv_caches[layer_name]  # 获取KV缓存
                
                # ==================== 更新注意力层布局 ====================
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:  # 如果是注意力规格且第一维为2
                    # ==================== 验证布局 ====================
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"  # 无法确定布局是(2, num_blocks, ...)还是(num_blocks, 2, ...)
                    )
                    
                    # ==================== 计算隐藏大小 ====================
                    hidden_size = kv_cache.shape[2:].numel()  # 计算隐藏大小
                    
                    # ==================== 更新步长 ====================
                    kv_cache.as_strided_(
                        size=kv_cache.shape,  # 保持形状不变
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),  # 更新步长
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        初始化KV缓存张量
        
        初始化KV缓存的内存缓冲区

        Args:
            kv_cache_config: KV缓存配置
            
        Returns:
            dict[str, torch.Tensor]: 层名到对应KV缓存内存缓冲区的映射
        """
        # ==================== 分配KV缓存原始张量 ====================
        # 初始化KV缓存的内存缓冲区
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)  # 分配KV缓存原始张量
        
        # ==================== 重塑KV缓存张量 ====================
        # 将内存缓冲区更改为所需的形状
        kv_caches = self._reshape_kv_cache_tensors(
            kv_cache_config, kv_cache_raw_tensors  # 重塑KV缓存张量
        )

        # ==================== 设置跨层KV缓存共享 ====================
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():  # 遍历共享KV缓存层
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)  # 记录调试信息
            kv_caches[layer_name] = kv_caches[target_layer_name]  # 设置层重用目标层的KV缓存

        # ==================== 绑定KV缓存 ====================
        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1  # 根据模型类型确定注意力模块数量
        )
        bind_kv_cache(
            kv_caches,  # KV缓存
            self.compilation_config.static_forward_context,  # 静态前向上下文
            self.kv_caches,  # KV缓存实例
            num_attn_module,  # 注意力模块数量
        )
        return kv_caches  # 返回KV缓存

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        可能添加KV共享层到KV缓存组
        
        将重用KV缓存的层添加到其目标层的KV缓存组中。
        KV缓存张量的映射在`initialize_kv_cache_tensors()`中发生
        """
        # ==================== 检查是否有共享KV缓存层 ====================
        if not self.shared_kv_cache_layers:  # 如果没有跨层KV共享
            # 没有跨层KV共享，返回
            return

        # ==================== 添加KV共享层到KV缓存组 ====================
        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,  # 共享KV缓存层
            kv_cache_config.kv_cache_groups,  # KV缓存组
            self.runner_only_attn_layers,  # 运行器专用注意力层
        )

        # ==================== 处理快速预填充 ====================
        if self.cache_config.kv_sharing_fast_prefill:  # 如果启用KV共享快速预填充
            # 在You Only Cache Once (https://arxiv.org/abs/2405.05254)或其他
            # 类似的KV共享设置中，只有生成KV缓存的层参与预填充阶段，
            # 使预填充能够提前退出
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)  # 获取注意力层
            for layer_name in reversed(attn_layers):  # 反向遍历注意力层
                if layer_name in self.shared_kv_cache_layers:  # 如果层在共享KV缓存层中
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)  # 添加到快速预填充合格层
                else:
                    break  # 否则跳出循环

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        初始化KV缓存
        
        基于`kv_cache_config`初始化KV缓存
        
        Args:
            kv_cache_config: KV缓存配置，包括每层的KV缓存大小
        """
        # ==================== 深拷贝配置 ====================
        kv_cache_config = deepcopy(kv_cache_config)  # 深拷贝KV缓存配置
        self.kv_cache_config = kv_cache_config  # 设置KV缓存配置
        
        # ==================== 添加编码器专用层 ====================
        self.may_add_encoder_only_layers_to_kv_cache_config()  # 可能添加编码器专用层到KV缓存配置
        
        # ==================== 添加KV共享层 ====================
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)  # 可能添加KV共享层到KV缓存组
        
        # ==================== 初始化注意力后端 ====================
        self.initialize_attn_backend(kv_cache_config)  # 初始化注意力后端
        
        # ==================== 重新初始化输入批次 ====================
        # 重新初始化需要在initialize_attn_backend之后
        self.may_reinitialize_input_batch(kv_cache_config)  # 可能重新初始化输入批次
        
        # ==================== 初始化KV缓存张量 ====================
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)  # 初始化KV缓存张量

        # ==================== 验证EAGLE推测配置 ====================
        if self.speculative_config and self.speculative_config.use_eagle():  # 如果有EAGLE推测配置
            assert isinstance(self.drafter, EagleProposer)  # 确保草稿器是EagleProposer
            # 验证所有草稿模型层属于同一个KV缓存组
            self.drafter.validate_same_kv_cache_group(kv_cache_config)  # 验证相同KV缓存组

        # ==================== 注册KV传输组 ====================
        if has_kv_transfer_group():  # 如果有KV传输组
            kv_transfer_group = get_kv_transfer_group()  # 获取KV传输组
            kv_transfer_group.register_kv_caches(kv_caches)  # 注册KV缓存
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)  # 设置主机传输缓冲区操作

        # ==================== 验证DCP配置 ====================
        if self.dcp_world_size > 1:  # 如果DCP世界大小大于1
            layer_names = self.attn_groups[0][0].layer_names  # 获取层名
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase, layer_names  # 获取注意力层
            )
            for layer in layers.values():  # 遍历层
                assert layer.impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer.impl.__class__.__name__} "
                    "does not return the softmax lse for decode."  # DCP要求注意力实现返回解码的softmax lse
                )

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        可能添加编码器专用层到KV缓存配置
        
        将编码器专用层添加到KV缓存配置中
        """
        # ==================== 获取配置参数 ====================
        block_size = self.vllm_config.cache_config.block_size  # 获取块大小
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)  # 初始化编码器专用注意力规格字典
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)  # 获取注意力层
        
        # ==================== 遍历注意力层 ====================
        for layer_name, attn_module in attn_layers.items():  # 遍历注意力层
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:  # 如果是编码器专用注意力类型
                # ==================== 创建编码器专用注意力规格 ====================
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,  # 块大小
                    num_kv_heads=attn_module.num_kv_heads,  # KV头数
                    head_size=attn_module.head_size,  # 头大小
                    dtype=self.kv_cache_dtype,  # 数据类型
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)  # 添加层名到规格
                self.runner_only_attn_layers.add(layer_name)  # 添加到运行器专用注意力层
        
        # ==================== 添加编码器专用层到KV缓存配置 ====================
        if len(encoder_only_attn_specs) > 0:  # 如果有编码器专用注意力规格
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"  # 目前只支持一个编码器专用注意力规格
            )
            spec, layer_names = encoder_only_attn_specs.popitem()  # 获取规格和层名
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)  # 添加KV缓存组规格
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        获取KV缓存规格
        
        通过解析静态前向上下文中每个注意力模块的KV缓存格式来生成KVCacheSpec
        
        Returns:
            dict[str, KVCacheSpec]: 层名到其KV缓存格式的字典映射。
            不需要KV缓存的层不包含在内
        """
        # ==================== 获取配置参数 ====================
        block_size = self.vllm_config.cache_config.block_size  # 获取块大小
        use_mla = self.vllm_config.model_config.use_mla  # 获取是否使用MLA
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype  # 获取缓存数据类型
        kv_cache_spec: dict[str, KVCacheSpec] = {}  # 初始化KV缓存规格字典
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)  # 获取注意力层
        
        # ==================== 遍历注意力层 ====================
        for layer_name, attn_module in attn_layers.items():  # 遍历注意力层
            # ==================== 处理注意力模块 ====================
            if isinstance(attn_module, Attention):  # 如果是注意力模块
                # ==================== 处理KV共享 ====================
                if (
                    kv_tgt_layer := attn_module.kv_sharing_target_layer_name  # 获取KV共享目标层名
                ) is not None:  # 如果存在KV共享目标层
                    # 该层不需要自己的KV缓存，将使用目标层的KV缓存。
                    # 我们跳过为其创建KVCacheSpec，这样KV缓存管理逻辑将
                    # 表现得好像该层不存在，不会为该层分配KV缓存。
                    # 这启用了跨层KV共享的内存节省，允许给定数量的内存
                    # 容纳更长的上下文长度或同时处理更多请求
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer  # 设置共享KV缓存层
                    continue  # 跳过

                # TODO(lucas): 将注意力规格移动到模型层中，就像注意力后端一样
                # ==================== 处理解码器注意力 ====================
                if attn_module.attn_type == AttentionType.DECODER:  # 如果是解码器注意力类型
                    # ==================== 处理滑动窗口注意力 ====================
                    if attn_module.sliding_window is not None:  # 如果有滑动窗口
                        assert not use_mla, "MLA is not supported for slidingwindow"  # MLA不支持滑动窗口
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,  # 块大小
                            num_kv_heads=attn_module.num_kv_heads,  # KV头数
                            head_size=attn_module.head_size,  # 头大小
                            dtype=self.kv_cache_dtype,  # 数据类型
                            sliding_window=attn_module.sliding_window,  # 滑动窗口
                        )
                    # ==================== 处理分块局部注意力 ====================
                    elif self.attention_chunk_size is not None and isinstance(
                        attn_module, ChunkedLocalAttention  # 如果是分块局部注意力
                    ):
                        kv_cache_spec[layer_name] = ChunkedLocalAttentionSpec(
                            block_size=block_size,  # 块大小
                            num_kv_heads=attn_module.num_kv_heads,  # KV头数
                            head_size=attn_module.head_size,  # 头大小
                            dtype=self.kv_cache_dtype,  # 数据类型
                            attention_chunk_size=self.attention_chunk_size,  # 注意力块大小
                        )
                    # ==================== 处理全注意力 ====================
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,  # 块大小
                            num_kv_heads=attn_module.num_kv_heads,  # KV头数
                            head_size=attn_module.head_size,  # 头大小
                            dtype=self.kv_cache_dtype,  # 数据类型
                        )
                # ==================== 处理编码器-解码器注意力 ====================
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:  # 如果是编码器-解码器注意力类型
                    kv_cache_spec[layer_name] = CrossAttentionSpec(
                        block_size=block_size,  # 块大小
                        num_kv_heads=attn_module.num_kv_heads,  # KV头数
                        head_size=attn_module.head_size,  # 头大小
                        dtype=self.kv_cache_dtype,  # 数据类型
                    )
                # ==================== 处理编码器专用注意力 ====================
                elif attn_module.attn_type in (
                    AttentionType.ENCODER,  # 编码器类型
                    AttentionType.ENCODER_ONLY,  # 编码器专用类型
                ):
                    # 编码器专用注意力不需要KV缓存
                    continue  # 跳过
                # ==================== 处理未知注意力类型 ====================
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")  # 未知注意力类型

            # ==================== 处理MLA注意力模块 ====================
            elif isinstance(attn_module, MLAAttention):  # 如果是MLA注意力模块
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,  # 块大小
                    num_kv_heads=1,  # KV头数为1
                    head_size=attn_module.head_size,  # 头大小
                    dtype=self.kv_cache_dtype,  # 数据类型
                    cache_dtype_str=cache_dtype_str,  # 缓存数据类型字符串
                )

            # ==================== 处理Mamba模块 ====================
            elif isinstance(attn_module, MambaBase):  # 如果是Mamba基础模块
                # ==================== 检查推测解码支持 ====================
                if (
                    self.vllm_config.speculative_config is not None  # 如果有推测配置
                    and self.vllm_config.model_config.hf_config.model_type  # 且模型类型
                    not in ["qwen3_next"]  # 不是qwen3_next
                ):
                    raise NotImplementedError(
                        "Mamba with speculative decoding is not supported yet."  # Mamba推测解码还不支持
                    )
                
                # ==================== 获取Mamba配置 ====================
                mamba_block_size = self.vllm_config.cache_config.mamba_block_size  # 获取Mamba块大小
                page_size_padded = self.vllm_config.cache_config.mamba_page_size_padded  # 获取填充页大小
                
                # ==================== 创建Mamba规格 ====================
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=attn_module.get_state_shape(),  # 获取状态形状
                    dtypes=attn_module.get_state_dtype(),  # 获取状态数据类型
                    block_size=mamba_block_size,  # 块大小
                    page_size_padded=page_size_padded,  # 填充页大小
                    mamba_type=attn_module.mamba_type,  # Mamba类型
                    num_speculative_blocks=(  # 推测块数量
                        self.speculative_config.num_speculative_tokens
                        if self.speculative_config  # 如果有推测配置
                        else 0  # 否则为0
                    ),
                )

        # ==================== 处理DeepSeek V32索引器缓存 ====================
        ds_indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, DeepseekV32IndexerCache  # 获取DeepSeek V32索引器缓存层
        )
        for layer_name, ds_indexer_module in ds_indexer_layers.items():  # 遍历索引器层
            kv_cache_spec[layer_name] = ds_indexer_module.get_kv_cache_spec()  # 获取KV缓存规格

        return kv_cache_spec  # 返回KV缓存规格

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        """
        将张量转换为列表
        
        将采样的token IDs张量转换为Python列表格式，
        使用CUDA事件同步来避免性能问题
        
        Args:
            sampled_token_ids: 采样的token IDs张量
            
        Returns:
            list[list[int]]: 转换后的token IDs列表
        """
        # ==================== 性能优化说明 ====================
        # 这是针对https://github.com/vllm-project/vllm/issues/22754中提到的问题的短期缓解措施。
        # `tolist`会触发CUDA流同步，这会阻塞其他CUDA流的复制操作。
        # CUDA事件同步可以避免这种情况。由于这在每个模型前向循环的关键路径中，
        # 这已经导致了分离设置中的性能问题。
        
        # ==================== 使用固定内存复制 ====================
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]  # 获取固定内存切片
        pinned.copy_(sampled_token_ids, non_blocking=True)  # 非阻塞复制到固定内存
        
        # ==================== 使用CUDA事件同步 ====================
        self.transfer_event.record()  # 记录传输事件
        self.transfer_event.synchronize()  # 同步事件，确保复制完成
        
        # ==================== 转换为列表 ====================
        return pinned.tolist()  # 将固定内存张量转换为列表
