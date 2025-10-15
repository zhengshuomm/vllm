# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    """
    调度器类
    
    负责管理请求的调度、KV缓存分配、推测解码等核心功能。
    这是vLLM V1引擎的核心调度组件。
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # ==================== 基础配置 ====================
        self.vllm_config = vllm_config  # vLLM配置
        self.scheduler_config = vllm_config.scheduler_config  # 调度器配置
        self.cache_config = vllm_config.cache_config  # 缓存配置
        self.lora_config = vllm_config.lora_config  # LoRA配置
        self.kv_cache_config = kv_cache_config  # KV缓存配置
        self.kv_events_config = vllm_config.kv_events_config  # KV事件配置
        self.parallel_config = vllm_config.parallel_config  # 并行配置
        self.log_stats = log_stats  # 是否记录统计信息
        self.structured_output_manager = structured_output_manager  # 结构化输出管理器
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder  # 是否为编码器-解码器模型

        # ==================== 完成请求跟踪 ====================
        # include_finished_set控制是否在update_from_outputs()返回的
        # EngineCoreOutputs中包含单独的已完成请求ID集合。
        # 这目前用于多引擎情况，以高效跟踪请求生命周期
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None  # 完成请求ID字典
        )

        # ==================== 调度约束 ====================
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # 最大运行请求数
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens  # 最大调度token数
        self.max_model_len = self.scheduler_config.max_model_len  # 最大模型长度
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events  # 是否启用KV缓存事件
        )

        # ==================== KV连接器 ====================
        # 为调度器创建KVConnector。注意每个Worker
        # 都有一个对应的Role=WORKER的KVConnector。
        # KV连接器用于P/D和卸载的远程KV推送/拉取
        self.connector = None  # KV连接器
        if self.vllm_config.kv_transfer_config is not None:  # 如果有KV传输配置
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors"  # 目前不支持多个KV缓存组的KV连接器
            )
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"  # 目前不支持编码器-解码器模型的KV连接器
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER  # 创建调度器KV连接器
            )

        # ==================== KV事件发布器 ====================
        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,  # KV事件配置
            self.parallel_config.data_parallel_rank,  # 数据并行排名
        )

        # ==================== 块大小配置 ====================
        num_gpu_blocks = self.cache_config.num_gpu_blocks  # GPU块数
        assert num_gpu_blocks is not None and num_gpu_blocks > 0  # 确保GPU块数有效

        self.block_size = self.cache_config.block_size  # 块大小

        # ==================== 分布式上下文并行 ====================
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size  # DCP世界大小
        # 注意(hc): 调度器的block_size必须乘以dcp_world_size，
        # 因为块哈希是在原始完整token序列上以
        # original_block_size × dcp_world_size的粒度计算的
        if self.dcp_world_size > 1:  # 如果DCP世界大小大于1
            self.block_size *= self.dcp_world_size  # 调整块大小

        # ==================== 请求管理 ====================
        # req_id -> Request
        self.requests: dict[str, Request] = {}  # 请求字典
        
        # ==================== 调度策略 ====================
        if self.scheduler_config.policy == "priority":  # 如果策略是优先级
            self.policy = SchedulingPolicy.PRIORITY  # 设置优先级策略
        elif self.scheduler_config.policy == "fcfs":  # 如果策略是先到先服务
            self.policy = SchedulingPolicy.FCFS  # 设置FCFS策略
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"  # 未知调度策略
            )
        
        # ==================== 请求队列 ====================
        # 请求的优先级队列
        self.waiting = create_request_queue(self.policy)  # 等待队列
        self.running: list[Request] = []  # 运行队列

        # ==================== 完成请求跟踪 ====================
        # 在前一步和当前步骤之间完成的请求ID。
        # 这用于通知工作器关于已完成的请求，
        # 以便它们可以释放这些请求的缓存状态。
        # 这在每个调度步骤结束时被刷新
        self.finished_req_ids: set[str] = set()  # 完成请求ID集合

        # ==================== KV连接器状态 ====================
        # KV连接器: 正在异步KV加载或接收过程中的请求
        self.finished_recving_kv_req_ids: set[str] = set()  # 完成接收KV的请求ID
        self.failed_recving_kv_req_ids: set[str] = set()  # 接收KV失败的请求ID

        # ==================== 编码器相关 ====================
        # 如果适用，计算编码器缓存大小
        # 注意: 目前我们对计算和空间使用相同的预算。
        # 当我们为跨请求的嵌入缓存制作编码器缓存时，这可以改变
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,  # 模型配置
            scheduler_config=vllm_config.scheduler_config,  # 调度器配置
            mm_registry=mm_registry,  # 多模态注册表
        )

        # 注意(woosuk): 这里，"编码器"包括MM模型的视觉编码器（和
        # 投影器，如果需要）以及编码器-解码器变换器
        self.max_num_encoder_input_tokens = encoder_compute_budget  # 最大编码器输入token数
        # 注意: 对于没有编码器的模型（例如，纯文本模型），
        # 编码器缓存不会被初始化，因为对于这些模型缓存大小为0
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)  # 编码器缓存管理器

        # ==================== 推测解码配置 ====================
        speculative_config = vllm_config.speculative_config  # 推测配置
        self.use_eagle = False  # 是否使用EAGLE
        self.num_spec_tokens = self.num_lookahead_tokens = 0  # 推测token数和前瞻token数
        if speculative_config:  # 如果有推测配置
            self.num_spec_tokens = speculative_config.num_speculative_tokens  # 推测token数
            if speculative_config.use_eagle():  # 如果使用EAGLE
                self.use_eagle = True  # 设置EAGLE标志
                self.num_lookahead_tokens = self.num_spec_tokens  # 设置前瞻token数

        # ==================== KV缓存管理器 ====================
        # 创建KV缓存管理器
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,  # KV缓存配置
            max_model_len=self.max_model_len,  # 最大模型长度
            enable_caching=self.cache_config.enable_prefix_caching,  # 是否启用前缀缓存
            use_eagle=self.use_eagle,  # 是否使用EAGLE
            log_stats=self.log_stats,  # 是否记录统计信息
            enable_kv_cache_events=self.enable_kv_cache_events,  # 是否启用KV缓存事件
            dcp_world_size=self.dcp_world_size,  # DCP世界大小
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1  # 是否使用流水线并行

    def schedule(self) -> SchedulerOutput:
        """
        调度请求
        
        这是调度器的核心方法，负责决定哪些请求应该被执行。
        
        注意(woosuk)关于调度算法:
        调度器中没有"解码阶段"或"预填充阶段"。
        每个请求只有num_computed_tokens和num_tokens_with_spec。
        num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)。
        在每一步中，调度器尝试为请求分配token，
        以便每个请求的num_computed_tokens能够赶上其num_tokens_with_spec。
        这足够通用，可以覆盖分块预填充、前缀缓存、推测解码，
        以及未来的"跳跃解码"优化。
        
        Returns:
            SchedulerOutput: 调度器输出，包含所有调度决策
        """
        # ==================== 初始化调度状态 ====================
        scheduled_new_reqs: list[Request] = []  # 新调度的请求
        scheduled_resumed_reqs: list[Request] = []  # 恢复的请求
        scheduled_running_reqs: list[Request] = []  # 继续运行的请求
        preempted_reqs: list[Request] = []  # 被抢占的请求

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}  # 请求到新块的映射
        num_scheduled_tokens: dict[str, int] = {}  # 调度的token数量
        token_budget = self.max_num_scheduled_tokens  # token预算
        
        # ==================== 编码器相关 ====================
        scheduled_encoder_inputs: dict[str, list[int]] = {}  # 调度的编码器输入
        encoder_compute_budget = self.max_num_encoder_input_tokens  # 编码器计算预算
        
        # ==================== 推测解码相关 ====================
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}  # 调度的推测解码token

        # ==================== 日志记录 ====================
        scheduled_timestamp = time.monotonic()  # 调度时间戳

        # ==================== 第一步：调度运行中的请求 ====================
        req_index = 0  # 请求索引
        while req_index < len(self.running) and token_budget > 0:  # 遍历运行中的请求
            request = self.running[req_index]  # 获取当前请求

            # ==================== 计算需要调度的新token数 ====================
            num_new_tokens = (
                request.num_tokens_with_spec  # 带推测的token数
                + request.num_output_placeholders  # 输出占位符数
                - request.num_computed_tokens  # 已计算的token数
            )
            # ==================== 长预填充处理 ====================
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:  # 如果超过长预填充阈值
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold  # 限制为阈值
            num_new_tokens = min(num_new_tokens, token_budget)  # 限制在预算内

            # ==================== 确保输入位置不超过最大模型长度 ====================
            # 在使用推测解码时这是必要的
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens  # 限制在最大模型长度内
            )

            # ==================== 调度编码器输入 ====================
            encoder_inputs_to_schedule = None  # 要调度的编码器输入
            new_encoder_compute_budget = encoder_compute_budget  # 新的编码器计算预算
            if request.has_encoder_inputs:  # 如果请求有编码器输入
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                ) = self._try_schedule_encoder_inputs(  # 尝试调度编码器输入
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            # ==================== 检查是否可以调度 ====================
            if num_new_tokens == 0:  # 如果没有新token要调度
                # 请求无法被调度，原因如下之一:
                # 1. 没有新token要调度。这可能发生在:
                #    (1) PP>1且我们已经调度了所有提示token但它们尚未完成
                #    (2) 异步调度且请求已达到其max_total_tokens或max_model_len
                # 2. 编码器预算已耗尽
                # 3. 编码器缓存已耗尽
                # 注意(woosuk): 这里通过使用`continue`而不是`break`，
                # 我们不严格遵循FCFS调度策略，允许较低优先级的请求被调度
                req_index += 1  # 移动到下一个请求
                continue  # 继续下一个请求

            # ==================== 为请求调度新需要的KV块 ====================
            while True:  # 循环直到分配成功
                new_blocks = self.kv_cache_manager.allocate_slots(  # 分配KV缓存槽位
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,  # 前瞻token数
                )

                if new_blocks is not None:  # 如果成功分配
                    # 请求可以被调度
                    break  # 跳出循环

                # ==================== 请求无法被调度，需要抢占 ====================
                # 抢占最低优先级的请求
                if self.policy == SchedulingPolicy.PRIORITY:  # 如果是优先级策略
                    preempted_req = max(  # 找到优先级最低的请求
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),  # 按优先级和到达时间排序
                    )
                    self.running.remove(preempted_req)  # 从运行队列移除
                    if preempted_req in scheduled_running_reqs:  # 如果在已调度列表中
                        scheduled_running_reqs.remove(preempted_req)  # 从已调度列表移除
                else:  # 如果是FCFS策略
                    preempted_req = self.running.pop()  # 弹出最后一个请求

                # ==================== 释放被抢占请求的资源 ====================
                self.kv_cache_manager.free(preempted_req)  # 释放KV缓存
                self.encoder_cache_manager.free(preempted_req)  # 释放编码器缓存
                preempted_req.status = RequestStatus.PREEMPTED  # 设置状态为被抢占
                preempted_req.num_computed_tokens = 0  # 重置已计算token数
                preempted_req.num_preemptions += 1  # 增加抢占次数
                if self.log_stats:  # 如果记录统计信息
                    preempted_req.record_event(
                        EngineCoreEventType.PREEMPTED, scheduled_timestamp  # 记录被抢占事件
                    )

                self.waiting.prepend_request(preempted_req)  # 将请求放回等待队列头部
                preempted_reqs.append(preempted_req)  # 添加到被抢占列表
                if preempted_req == request:  # 如果被抢占的是当前请求
                    # 没有更多请求可以抢占。无法调度此请求
                    break  # 跳出循环

            if new_blocks is None:  # 如果仍然无法分配
                # 无法调度此请求
                break  # 跳出循环

            # ==================== 调度请求 ====================
            scheduled_running_reqs.append(request)  # 添加到已调度运行列表
            req_to_new_blocks[request.request_id] = new_blocks  # 记录新块
            num_scheduled_tokens[request.request_id] = num_new_tokens  # 记录调度token数
            token_budget -= num_new_tokens  # 减少token预算
            req_index += 1  # 移动到下一个请求

            # ==================== 推测解码相关 ====================
            if request.spec_token_ids:  # 如果有推测token
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens  # 计算调度的推测token数
                )
                if num_scheduled_spec_tokens > 0:  # 如果有推测token要调度
                    # 将spec_token_ids列表修剪到num_scheduled_spec_tokens
                    del request.spec_token_ids[num_scheduled_spec_tokens:]  # 删除多余的推测token
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids  # 记录调度的推测解码token
                    )

            # ==================== 编码器相关 ====================
            if encoder_inputs_to_schedule:  # 如果有编码器输入要调度
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule  # 记录调度的编码器输入
                )
                # 分配编码器缓存
                for i in encoder_inputs_to_schedule:  # 遍历编码器输入
                    self.encoder_cache_manager.allocate(request, i)  # 分配编码器缓存
                encoder_compute_budget = new_encoder_compute_budget  # 更新编码器计算预算

        # ==================== 记录已调度运行请求中的LoRA ====================
        scheduled_loras: set[int] = set()  # 已调度的LoRA集合
        if self.lora_config:  # 如果有LoRA配置
            scheduled_loras = set(
                req.lora_request.lora_int_id  # 获取LoRA整数ID
                for req in scheduled_running_reqs  # 遍历已调度的运行请求
                if req.lora_request and req.lora_request.lora_int_id > 0  # 如果有LoRA请求且ID大于0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras  # 确保不超过最大LoRA数

        # ==================== 创建临时请求队列 ====================
        # 使用临时RequestQueue收集需要跳过的请求，
        # 稍后将其放回等待队列的头部
        skipped_waiting_requests = create_request_queue(self.policy)  # 创建跳过的等待请求队列

        # ==================== 第二步：调度等待中的请求 ====================
        if not preempted_reqs:  # 如果没有被抢占的请求
            while self.waiting and token_budget > 0:  # 遍历等待队列
                if len(self.running) == self.max_num_running_reqs:  # 如果运行队列已满
                    break  # 跳出循环

                request = self.waiting.peek_request()  # 查看队列头部请求

                # ==================== KV传输：跳过仍在等待远程KV的请求 ====================
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 如果状态是等待远程KV
                    is_ready = self._update_waiting_for_remote_kv(request)  # 更新等待远程KV状态
                    if is_ready:  # 如果准备就绪
                        request.status = RequestStatus.WAITING  # 设置为等待状态
                    else:  # 如果未准备就绪
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,  # 记录调试信息
                        )
                        self.waiting.pop_request()  # 弹出请求
                        skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                        continue  # 继续下一个请求

                # ==================== 跳过仍在等待FSM编译的结构化输出请求 ====================
                if request.status == RequestStatus.WAITING_FOR_FSM:  # 如果状态是等待FSM
                    structured_output_req = request.structured_output_request  # 获取结构化输出请求
                    if structured_output_req and structured_output_req.grammar:  # 如果有语法
                        request.status = RequestStatus.WAITING  # 设置为等待状态
                    else:  # 如果没有语法
                        self.waiting.pop_request()  # 弹出请求
                        skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                        continue  # 继续下一个请求

                # ==================== 检查添加请求是否仍遵守max_loras约束 ====================
                if (
                    self.lora_config  # 如果有LoRA配置
                    and request.lora_request  # 且请求有LoRA请求
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras  # 且已调度LoRA数达到最大
                        and request.lora_request.lora_int_id not in scheduled_loras  # 且当前请求的LoRA不在已调度中
                    )
                ):
                    # 调度将超过max_loras，跳过
                    self.waiting.pop_request()  # 弹出请求
                    skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                    continue  # 继续下一个请求

                # ==================== 初始化变量 ====================
                num_external_computed_tokens = 0  # 外部计算的token数
                load_kv_async = False  # 是否异步加载KV

                # ==================== 获取已缓存的token ====================
                if request.num_computed_tokens == 0:  # 如果已计算token数为0
                    # 获取本地缓存的token
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)  # 获取已计算块
                    )

                    # ==================== 如果使用KVConnector，获取外部缓存的token ====================
                    if self.connector is not None:  # 如果有连接器
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(  # 获取匹配的token数
                                request, num_new_local_computed_tokens
                            )
                        )

                        if num_external_computed_tokens is None:  # 如果无法确定外部token数
                            # 请求无法被调度，因为KVConnector无法确定匹配token数
                            self.waiting.pop_request()  # 弹出请求
                            skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                            continue  # 继续下一个请求

                    # 总计算token数（本地+外部）
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens  # 计算总token数
                    )
                # ==================== KV传输：等待请求在异步KV接收完成后num_computed_tokens > 0 ====================
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list()  # 创建空块列表
                    )
                    num_new_local_computed_tokens = 0  # 新本地计算token数为0
                    num_computed_tokens = request.num_computed_tokens  # 使用请求的已计算token数

                # ==================== 初始化编码器相关变量 ====================
                encoder_inputs_to_schedule = None  # 要调度的编码器输入
                new_encoder_compute_budget = encoder_compute_budget  # 新的编码器计算预算

                # ==================== KV传输：加载远程KV，不为新工作分配 ====================
                if load_kv_async:  # 如果异步加载KV
                    assert num_external_computed_tokens > 0  # 确保外部计算token数大于0
                    num_new_tokens = 0  # 新token数为0
                # ==================== 要调度的token数量 ====================
                else:
                    # 我们使用`request.num_tokens`而不是`request.num_prompt_tokens`
                    # 来考虑恢复的请求，它们有输出token
                    num_new_tokens = request.num_tokens - num_computed_tokens  # 计算新token数
                    if (
                        0
                        < self.scheduler_config.long_prefill_token_threshold  # 如果长预填充阈值大于0
                        < num_new_tokens  # 且新token数超过阈值
                    ):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold  # 限制为阈值
                        )

                    # ==================== 分块预填充检查 ====================
                    # 分块预填充必须显式启用才能允许池化请求被分块
                    if (
                        not self.scheduler_config.chunked_prefill_enabled  # 如果未启用分块预填充
                        and num_new_tokens > token_budget  # 且新token数超过预算
                    ):
                        self.waiting.pop_request()  # 弹出请求
                        skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                        continue  # 继续下一个请求

                    num_new_tokens = min(num_new_tokens, token_budget)  # 限制在预算内
                    assert num_new_tokens > 0  # 确保新token数大于0

                    # ==================== 调度编码器输入 ====================
                    if request.has_encoder_inputs:  # 如果请求有编码器输入
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                        ) = self._try_schedule_encoder_inputs(  # 尝试调度编码器输入
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                        )
                        if num_new_tokens == 0:  # 如果新token数为0
                            # 请求无法被调度
                            break  # 跳出循环

                # ==================== 处理P/D分离与推测解码的边缘情况 ====================
                # 处理P/D分离与推测解码一起使用时的边缘情况，
                # 其中会分配额外的块，导致本地和远程块数量不匹配
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens  # 有效前瞻token数
                )

                # ==================== 确定是否需要分配交叉注意力块 ====================
                if self.is_encoder_decoder and request.has_encoder_inputs:  # 如果是编码器-解码器且有编码器输入
                    # TODO(russellb): 对于Whisper，我们知道输入总是填充到最大长度。
                    # 如果我们支持其他编码器-解码器模型，如果我们只想分配需要的部分，
                    # 这需要更新
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens  # 编码器token数
                    )
                else:
                    num_encoder_tokens = 0  # 编码器token数为0

                # ==================== 分配KV缓存槽位 ====================
                new_blocks = self.kv_cache_manager.allocate_slots(  # 分配KV缓存槽位
                    request,
                    num_new_tokens + num_external_computed_tokens,  # 新token数+外部计算token数
                    num_new_local_computed_tokens,  # 新本地计算token数
                    new_computed_blocks,  # 新计算块
                    num_lookahead_tokens=effective_lookahead_tokens,  # 有效前瞻token数
                    delay_cache_blocks=load_kv_async,  # 是否延迟缓存块
                    num_encoder_tokens=num_encoder_tokens,  # 编码器token数
                )

                if new_blocks is None:  # 如果无法分配
                    # 请求无法被调度
                    break  # 跳出循环

                # ==================== KV传输：连接器使用此信息确定是否需要加载 ====================
                # 注意：此信息用于确定此请求是否需要加载
                if self.connector is not None:  # 如果有连接器
                    self.connector.update_state_after_alloc(  # 分配后更新状态
                        request,
                        new_computed_blocks + new_blocks,  # 新计算块+新块
                        num_external_computed_tokens,  # 外部计算token数
                    )

                # ==================== 处理请求 ====================
                # 请求已经从self.waiting中弹出，
                # 除非由于new_blocks为None而重新添加
                request = self.waiting.pop_request()  # 弹出请求
                if load_kv_async:  # 如果异步加载
                    # 如果异步加载，分配内存并将请求放入WAITING_FOR_REMOTE_KV状态
                    skipped_waiting_requests.prepend_request(request)  # 添加到跳过队列
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS  # 设置状态
                    continue  # 继续下一个请求

                # ==================== 将请求添加到运行队列 ====================
                req_index += 1  # 增加请求索引
                self.running.append(request)  # 添加到运行队列
                if self.log_stats:  # 如果记录统计信息
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp  # 记录调度事件
                    )
                if request.status == RequestStatus.WAITING:  # 如果状态是等待
                    scheduled_new_reqs.append(request)  # 添加到新调度列表
                elif request.status == RequestStatus.PREEMPTED:  # 如果状态是被抢占
                    scheduled_resumed_reqs.append(request)  # 添加到恢复列表
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")  # 无效请求状态

                # ==================== 更新LoRA和块信息 ====================
                if self.lora_config and request.lora_request:  # 如果有LoRA配置和请求
                    scheduled_loras.add(request.lora_request.lora_int_id)  # 添加LoRA ID
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)  # 获取请求的块
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens  # 记录调度token数
                token_budget -= num_new_tokens  # 减少token预算
                request.status = RequestStatus.RUNNING  # 设置状态为运行
                request.num_computed_tokens = num_computed_tokens  # 设置已计算token数
                # ==================== 计算前缀缓存token数 ====================
                if request.num_cached_tokens < 0:  # 如果缓存token数小于0
                    request.num_cached_tokens = num_computed_tokens  # 设置为已计算token数
                # ==================== 编码器相关 ====================
                if encoder_inputs_to_schedule:  # 如果有编码器输入要调度
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule  # 记录编码器输入
                    )
                    # 分配编码器缓存
                    for i in encoder_inputs_to_schedule:  # 遍历编码器输入
                        self.encoder_cache_manager.allocate(request, i)  # 分配编码器缓存
                    encoder_compute_budget = new_encoder_compute_budget  # 更新编码器计算预算

        # ==================== 将跳过的请求放回等待队列头部 ====================
        if skipped_waiting_requests:  # 如果有跳过的请求
            self.waiting.prepend_requests(skipped_waiting_requests)  # 将请求放回等待队列头部

        # ==================== 检查调度约束是否满足 ====================
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())  # 计算总调度token数
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens  # 确保不超过最大调度token数
        assert token_budget >= 0  # 确保token预算非负
        assert len(self.running) <= self.max_num_running_reqs  # 确保运行请求数不超过最大限制
        # 由于运行队列中的某些请求可能在此步骤中未被调度，
        # 已调度请求的总数可能小于len(self.running)
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs  # 确保已调度请求数不超过运行队列长度
        ) <= len(self.running)

        # ==================== 获取运行队列中所有请求的最长公共前缀 ====================
        # 这可以用于级联注意力
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)  # 初始化公共前缀块数
        if self.running:  # 如果有运行中的请求
            any_request = self.running[0]  # 获取任意一个请求
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(  # 获取公共前缀块数
                    any_request.request_id
                )
            )

        # ==================== 构建调度器输出 ====================
        new_reqs_data = [
            NewRequestData.from_request(  # 从请求创建新请求数据
                req, req_to_new_blocks[req.request_id].get_block_ids()  # 获取块ID
            )
            for req in scheduled_new_reqs  # 遍历新调度的请求
        ]
        cached_reqs_data = self._make_cached_request_data(  # 创建缓存请求数据
            scheduled_running_reqs,  # 已调度的运行请求
            scheduled_resumed_reqs,  # 已调度的恢复请求
            num_scheduled_tokens,  # 调度token数
            scheduled_spec_decode_tokens,  # 调度的推测解码token
            req_to_new_blocks,  # 请求到新块的映射
        )
        scheduled_requests = (
            scheduled_new_reqs + scheduled_running_reqs + scheduled_resumed_reqs  # 所有已调度的请求
        )
        structured_output_request_ids, grammar_bitmask = self.get_grammar_bitmask(  # 获取语法位掩码
            scheduled_requests, scheduled_spec_decode_tokens
        )
        scheduler_output = SchedulerOutput(  # 创建调度器输出
            scheduled_new_reqs=new_reqs_data,  # 新调度的请求
            scheduled_cached_reqs=cached_reqs_data,  # 已调度的缓存请求
            num_scheduled_tokens=num_scheduled_tokens,  # 调度token数
            total_num_scheduled_tokens=total_num_scheduled_tokens,  # 总调度token数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,  # 调度的推测解码token
            scheduled_encoder_inputs=scheduled_encoder_inputs,  # 调度的编码器输入
            num_common_prefix_blocks=num_common_prefix_blocks,  # 公共前缀块数
            # finished_req_ids是调度器中的现有状态，
            # 而不是在此步骤中新调度的。
            # 它包含在前一步和当前步骤之间完成的请求ID
            finished_req_ids=self.finished_req_ids,  # 完成的请求ID
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),  # 释放的编码器多模态哈希
            structured_output_request_ids=structured_output_request_ids,  # 结构化输出请求ID
            grammar_bitmask=grammar_bitmask,  # 语法位掩码
        )

        # ==================== 构建KV连接器元数据 ====================
        # 注意(Kuntai): 此函数设计用于多个目的:
        # 1. 规划KV缓存存储
        # 2. 将所有KV缓存加载/保存操作包装到不透明对象中
        # 3. 清除连接器的内部状态
        if self.connector is not None:  # 如果有连接器
            meta = self.connector.build_connector_meta(scheduler_output)  # 构建连接器元数据
            scheduler_output.kv_connector_metadata = meta  # 设置KV连接器元数据

        # ==================== 收集KV缓存事件 ====================
        # 从KV缓存管理器收集KV缓存事件
        events = self.kv_cache_manager.take_events()  # 获取KV缓存事件

        # 从连接器收集KV缓存事件
        if self.connector is not None:  # 如果有连接器
            connector_events = self.connector.take_events()  # 获取连接器事件
            if connector_events:  # 如果有连接器事件
                if events is None:  # 如果事件为None
                    events = list(connector_events)  # 创建事件列表
                else:
                    events.extend(connector_events)  # 扩展事件列表

        # ==================== 发布收集的KV缓存事件 ====================
        if events:  # 如果有事件
            batch = KVEventBatch(ts=time.time(), events=events)  # 创建事件批次
            self.kv_event_publisher.publish(batch)  # 发布事件批次

        # ==================== 调度后更新 ====================
        self._update_after_schedule(scheduler_output)  # 调度后更新
        return scheduler_output  # 返回调度器输出

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """
        调度后更新
        
        在请求被调度后推进请求的已计算token数。
        
        1. 当前步骤的scheduler_output必须包含原始调度token数来确定输入ID
        2. 在这里推进已计算token数，允许我们在下一个调度步骤中立即重新调度预填充请求
        3. 如果某些token（例如推测token）稍后被拒绝，已计算token数将在update_from_output中调整
        """
        # ==================== 推进已计算token数 ====================
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 获取调度token数
        for req_id, num_scheduled_token in num_scheduled_tokens.items():  # 遍历调度token数
            request = self.requests[req_id]  # 获取请求
            request.num_computed_tokens += num_scheduled_token  # 增加已计算token数

            # ==================== 释放编码器输入 ====================
            # 注意: _free_encoder_inputs依赖于num_computed_tokens，
            # 它可能在_update_from_output中因推测解码而再次更新。
            # 但是在这里调用该方法是安全的，因为编码器输入总是提示的一部分，
            # 而不是输出，因此不受推测解码的影响
            if request.has_encoder_inputs:  # 如果请求有编码器输入
                self._free_encoder_inputs(request)  # 释放编码器输入

        # ==================== 清除完成的请求ID ====================
        # 注意: 我们不应该在这里执行self.finished_req_ids.clear()，
        # 因为它也会影响调度器输出
        self.finished_req_ids = set()  # 重置完成的请求ID集合

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        """
        创建缓存请求数据
        
        为运行中和恢复的请求创建缓存请求数据，包含token ID、块ID等信息
        
        Args:
            running_reqs: 运行中的请求列表
            resumed_reqs: 恢复的请求列表
            num_scheduled_tokens: 调度token数映射
            spec_decode_tokens: 推测解码token映射
            req_to_new_blocks: 请求到新块的映射
            
        Returns:
            CachedRequestData: 缓存请求数据
        """
        # ==================== 初始化数据列表 ====================
        req_ids: list[str] = []  # 请求ID列表
        new_token_ids: list[list[int]] = []  # 新token ID列表
        new_block_ids: list[tuple[list[int], ...] | None] = []  # 新块ID列表
        resumed_req_token_ids: list[list[int] | None] = []  # 恢复请求token ID列表
        num_computed_tokens: list[int] = []  # 已计算token数列表
        num_output_tokens: list[int] = []  # 输出token数列表

        # ==================== 创建抢占恢复标志 ====================
        # 因为resumed_reqs通常为空，就地追加更高效，
        # 这样我们不需要分配新列表
        resumed_from_preemption = [False] * len(running_reqs)  # 运行请求的抢占标志
        resumed_from_preemption += [True] * len(resumed_reqs)  # 恢复请求的抢占标志
        
        # ==================== 遍历所有请求 ====================
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):  # 遍历运行和恢复请求
            req_id = req.request_id  # 获取请求ID
            req_ids.append(req_id)  # 添加请求ID
            
            # ==================== 计算token数 ====================
            num_tokens = num_scheduled_tokens[req_id] - len(
                spec_decode_tokens.get(req_id, ())  # 计算实际token数（排除推测token）
            )
            
            # ==================== 处理流水线并行 ====================
            if self.use_pp:  # 如果使用流水线并行
                # 当使用PP时，调度器发送采样的token回来，
                # 因为第一阶段工作器和最后阶段工作器之间没有直接通信。
                # 否则，我们不需要发送采样的token回来，因为模型运行器会缓存它们
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens  # 获取token ID
                ]
                new_token_ids.append(token_ids)  # 添加新token ID
            
            # ==================== 处理恢复请求 ====================
            resumed_token_ids = None  # 初始化恢复token ID
            if resumed_from_preemption[idx]:  # 如果从抢占恢复
                resumed_token_ids = req.all_token_ids[
                    : req.num_computed_tokens + num_tokens  # 获取恢复token ID
                ]
            resumed_req_token_ids.append(resumed_token_ids)  # 添加恢复token ID
            
            # ==================== 添加块和token信息 ====================
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)  # 获取新块ID
            )
            num_computed_tokens.append(req.num_computed_tokens)  # 添加已计算token数
            num_output_tokens.append(req.num_output_tokens)  # 添加输出token数

        # ==================== 返回缓存请求数据 ====================
        return CachedRequestData(
            req_ids=req_ids,  # 请求ID
            resumed_from_preemption=resumed_from_preemption,  # 从抢占恢复标志
            new_token_ids=new_token_ids,  # 新token ID
            resumed_req_token_ids=resumed_req_token_ids,  # 恢复请求token ID
            new_block_ids=new_block_ids,  # 新块ID
            num_computed_tokens=num_computed_tokens,  # 已计算token数
            num_output_tokens=num_output_tokens,  # 输出token数
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        尝试调度编码器输入
        
        确定在当前步骤中需要调度哪些编码器输入，
        并相应地更新`num_new_tokens`和编码器token预算。
        
        编码器输入将在以下情况下被调度:
        - 其输出token与当前步骤中正在计算的token范围重叠，即
          [num_computed_tokens, num_computed_tokens + num_new_tokens)
        - 尚未计算并存储在编码器缓存中
        - 有足够的编码器token预算来处理它
        - 编码器缓存有空间存储它
        
        如果由于缓存或预算限制无法调度编码器输入，
        该方法调整`num_new_tokens`以仅调度到不可调度编码器输入之前的解码器token。
        
        注意num_computed_tokens包括本地缓存的块和外部缓存的块（通过KVConnector）。
        
        Args:
            request: 请求对象
            num_computed_tokens: 已计算的token数
            num_new_tokens: 新token数
            encoder_compute_budget: 编码器计算预算
            
        Returns:
            tuple: (要调度的编码器输入列表, 调整后的新token数, 调整后的编码器计算预算)
        """
        # ==================== 早期返回检查 ====================
        if num_new_tokens == 0 or not request.has_encoder_inputs:  # 如果没有新token或没有编码器输入
            return [], num_new_tokens, encoder_compute_budget  # 返回空列表
        
        # ==================== 初始化变量 ====================
        encoder_inputs_to_schedule: list[int] = []  # 要调度的编码器输入列表
        mm_features = request.mm_features  # 多模态特征
        assert mm_features is not None  # 确保多模态特征不为None
        assert len(mm_features) > 0  # 确保多模态特征不为空

        # ==================== 创建临时跟踪器 ====================
        # 注意: 由于调度器在请求级别操作（每个请求可能有多个编码器输入），
        # 我们需要创建临时跟踪器以在编码器输入级别进行记账
        mm_hashes_to_schedule = set()  # 要调度的多模态哈希集合
        num_tokens_to_schedule = 0  # 要调度的token数
        
        # ==================== 遍历多模态特征 ====================
        for i, mm_feature in enumerate(mm_features):  # 遍历多模态特征
            start_pos = mm_feature.mm_position.offset  # 获取起始位置
            num_encoder_tokens = mm_feature.mm_position.length  # 获取编码器token数

            # ==================== 检查编码器输出是否需要 ====================
            # 如果两个范围重叠，则需要编码器输出:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) 和
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:  # 如果起始位置超出范围
                # 此步骤不需要编码器输入
                break  # 跳出循环

            # ==================== 处理编码器-解码器模型 ====================
            if self.is_encoder_decoder and num_computed_tokens > 0:  # 如果是编码器-解码器模型且已计算token数大于0
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."  # 编码器输入应在序列开始时处理
                )
                # 编码器输入已经计算
                # 这里的计算有点不同。我们不将编码器输出转换为
                # 由解码器处理并反映在num_computed_tokens中的token。
                # 相反，start_pos反映我们需要确保计算编码器输入的位置。
                # 这应该总是0，以确保我们在运行解码器之前计算编码器输入。
                # 一旦我们计算了一些解码器token（num_computed_tokens > 0），
                # 我们就知道我们已经计算了编码器输入，可以在这里跳过
                continue  # 继续下一个特征

            elif start_pos + num_encoder_tokens <= num_computed_tokens:  # 如果编码器输入已经计算
                # 编码器输入已经计算并存储在解码器的KV缓存中
                continue  # 继续下一个特征

            # ==================== 处理非编码器-解码器模型 ====================
            if not self.is_encoder_decoder:  # 如果不是编码器-解码器模型
                # 我们还没有为编码器-解码器模型使用编码器缓存
                if request.mm_features[i].identifier in mm_hashes_to_schedule:  # 如果标识符已在调度中
                    # 相同的编码器输入已在当前步骤中调度
                    continue  # 继续下一个特征

                if self.encoder_cache_manager.check_and_update_cache(request, i):  # 如果检查并更新缓存
                    # 编码器输入已经计算并从之前的步骤缓存
                    continue  # 继续下一个特征

            # ==================== 检查分块多模态输入 ====================
            # 如果不允许编码器输入分块，我们不想部分调度多模态项目。
            # 如果调度范围只覆盖多模态输入的一部分，回滚到多模态项目之前
            if (
                self.scheduler_config.disable_chunked_mm_input  # 如果禁用分块多模态输入
                and num_computed_tokens < start_pos  # 且已计算token数小于起始位置
                and (num_computed_tokens + num_new_tokens)  # 且调度范围
                < (start_pos + num_encoder_tokens)  # 只覆盖多模态输入的一部分
            ):
                num_new_tokens = start_pos - num_computed_tokens  # 调整新token数
                break  # 跳出循环

            # ==================== 检查编码器缓存分配 ====================
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_tokens_to_schedule  # 检查是否可以分配
            ):
                # 编码器缓存已满或编码器预算已耗尽
                # 注意(woosuk): 我们假设编码器输入token应该一起处理，
                # 因为编码器通常使用双向注意力
                if num_computed_tokens < start_pos:  # 如果已计算token数小于起始位置
                    # 我们只调度编码器输入之前的解码器token
                    num_new_tokens = start_pos - num_computed_tokens  # 调整新token数
                else:  # 否则
                    # 由于前缀缓存，num_computed_tokens大于start_pos，
                    # 即使其编码器输入不可用。在这种情况下，
                    # 我们无法在此步骤中为此请求调度任何token
                    num_new_tokens = 0  # 设置新token数为0
                break  # 跳出循环

            # ==================== 更新调度信息 ====================
            num_tokens_to_schedule += num_encoder_tokens  # 增加要调度的token数
            encoder_compute_budget -= num_encoder_tokens  # 减少编码器计算预算
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)  # 添加多模态哈希
            encoder_inputs_to_schedule.append(i)  # 添加编码器输入索引

        # ==================== 返回结果 ====================
        return (
            encoder_inputs_to_schedule,  # 要调度的编码器输入列表
            num_new_tokens,  # 调整后的新token数
            encoder_compute_budget,  # 调整后的编码器计算预算
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        """
        获取语法位掩码
        
        为使用结构化输出的请求生成语法位掩码，用于约束token生成。
        
        Args:
            requests: 请求列表
            scheduled_spec_decode_tokens: 调度的推测解码token映射
            
        Returns:
            tuple: (结构化输出请求ID映射, 语法位掩码)
        """
        # ==================== 构建结构化输出请求ID映射 ====================
        # structured_output_request_ids 将使用结构化输出的请求的request_id
        # 映射到其在批次中的索引。这有助于我们确定如何切片语法位掩码，
        # 并仅对使用结构化解码的请求应用有效掩码。
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:  # 如果请求使用结构化输出
                # 性能优化: 在分块预填充的情况下，
                # 请求可能不包含任何新token。
                # 因此，我们可能会引入一些额外的
                # 周期来填充位掩码，这可能是一个大的无操作。
                structured_output_request_ids[req.request_id] = i  # 记录请求ID和索引

        # ==================== 生成语法位掩码 ====================
        if not structured_output_request_ids:  # 如果没有结构化输出请求
            bitmask = None  # 位掩码为None
        else:  # 如果有结构化输出请求
            bitmask = self.structured_output_manager.grammar_bitmask(  # 生成语法位掩码
                self.requests,  # 所有请求
                structured_output_request_ids,  # 结构化输出请求ID映射
                scheduled_spec_decode_tokens,  # 调度的推测解码token
            )
        return structured_output_request_ids, bitmask  # 返回映射和位掩码

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """
        从模型输出更新调度器状态
        
        处理模型运行器的输出，更新请求状态，处理停止条件，
        并生成引擎核心输出。
        
        Args:
            scheduler_output: 调度器输出
            model_runner_output: 模型运行器输出
            
        Returns:
            dict[int, EngineCoreOutputs]: 按客户端索引分组的引擎核心输出
        """
        # ==================== 提取模型输出数据 ====================
        sampled_token_ids = model_runner_output.sampled_token_ids  # 采样的token ID
        logprobs = model_runner_output.logprobs  # 对数概率
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict  # 提示对数概率字典
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens  # 调度的token数
        pooler_outputs = model_runner_output.pooler_output  # 池化器输出
        num_nans_in_logits = model_runner_output.num_nans_in_logits  # logits中的NaN数量
        kv_connector_output = model_runner_output.kv_connector_output  # KV连接器输出

        # ==================== 初始化输出和统计 ====================
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)  # 按客户端分组的输出
        spec_decoding_stats: SpecDecodingStats | None = None  # 推测解码统计
        kv_connector_stats = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None  # KV连接器统计
        )

        # ==================== 处理KV加载失败 ====================
        failed_kv_load_req_ids = None  # 失败的KV加载请求ID
        if kv_connector_output and kv_connector_output.invalid_block_ids:  # 如果有无效块ID
            # 这些块包含加载失败的外部计算token。
            # 识别受影响的请求并调整其已计算token数
            # 以触发无效块的重新计算。
            failed_kv_load_req_ids = self._handle_invalid_blocks(  # 处理无效块
                kv_connector_output.invalid_block_ids  # 无效块ID
            )

        # ==================== 处理每个调度的请求 ====================
        # 注意(woosuk): 由于len(num_scheduled_tokens)可能达到1K或更多，
        # 下面的循环可能成为性能瓶颈。我们应该尽力
        # 避免在循环内进行昂贵的操作。
        stopped_running_reqs: set[Request] = set()  # 停止的运行请求
        stopped_preempted_reqs: set[Request] = set()  # 停止的被抢占请求
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():  # 遍历调度的token数
            assert num_tokens_scheduled > 0  # 确保调度的token数大于0
            
            # ==================== 跳过KV加载失败的请求 ====================
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:  # 如果请求在失败列表中
                # 跳过从KV加载失败中恢复的请求
                continue  # 继续下一个请求
            
            # ==================== 获取请求对象 ====================
            request = self.requests.get(req_id)  # 获取请求对象
            if request is None:  # 如果请求为None
                # 请求已经完成。这可能发生在
                # 请求在模型执行时被中止（例如，
                # 在流水线并行中）。
                continue  # 继续下一个请求

            # ==================== 获取生成的token ====================
            req_index = model_runner_output.req_id_to_index[req_id]  # 获取请求索引
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []  # 获取生成的token ID
            )

            # ==================== 处理推测解码 ====================
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)  # 获取调度的推测token
            )
            if scheduled_spec_token_ids:  # 如果有推测token
                num_draft_tokens = len(scheduled_spec_token_ids)  # 草稿token数
                num_accepted = len(generated_token_ids) - 1  # 接受的token数
                num_rejected = num_draft_tokens - num_accepted  # 拒绝的token数
                # num_computed_tokens表示在当前步骤中处理的token数，
                # 考虑调度的token和拒绝。如果某些token被拒绝，
                # num_computed_tokens会减少被拒绝的token数。
                request.num_computed_tokens -= num_rejected  # 减少已计算token数
                spec_decoding_stats = self.make_spec_decoding_stats(  # 创建推测解码统计
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,  # 草稿token数
                    num_accepted_tokens=num_accepted,  # 接受的token数
                )

            # ==================== 初始化变量 ====================
            stopped = False  # 是否停止
            new_logprobs = None  # 新对数概率
            new_token_ids = generated_token_ids  # 新token ID
            kv_transfer_params = None  # KV传输参数
            status_before_stop = request.status  # 停止前的状态

            # ==================== 检查停止条件并更新请求状态 ====================
            if new_token_ids:  # 如果有新token
                new_token_ids, stopped = self._update_request_with_output(  # 更新请求输出
                    request, new_token_ids
                )

            # ==================== 池化器模型的停止检查 ====================
            pooler_output = None  # 池化器输出
            if pooler_outputs:  # 如果有池化器输出
                pooler_output = pooler_outputs[req_index]  # 获取池化器输出
                stopped = check_stop(request, self.max_model_len, pooler_output)  # 检查停止条件

            # ==================== 处理停止的请求 ====================
            if stopped:  # 如果请求停止
                kv_transfer_params = self._free_request(request)  # 释放请求
                if status_before_stop == RequestStatus.RUNNING:  # 如果之前是运行状态
                    stopped_running_reqs.add(request)  # 添加到停止的运行请求
                else:  # 否则
                    stopped_preempted_reqs.add(request)  # 添加到停止的被抢占请求

            # ==================== 提取采样对数概率 ====================
            if (
                request.sampling_params is not None  # 如果有采样参数
                and request.sampling_params.logprobs is not None  # 且需要对数概率
                and logprobs  # 且有对数概率数据
            ):
                # 注意: 一旦我们支持每步N个token（推测解码），
                # 外部列表的长度可能大于1。
                new_logprobs = logprobs.slice(req_index, req_index + 1)  # 切片对数概率

            # ==================== 处理结构化输出 ====================
            if new_token_ids and self.structured_output_manager.should_advance(request):  # 如果有新token且应该推进
                # 注意: 如果use_structured_output为True，
                # structured_output_request不应该为None，我们已经在
                # 上面检查过了，所以可以安全地忽略类型警告
                request.structured_output_request.grammar.accept_tokens(  # 接受token
                    req_id, new_token_ids
                )

            # ==================== 处理NaN检查 ====================
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:  # 如果有NaN检查
                request.num_nans_in_logits = num_nans_in_logits[req_id]  # 设置NaN数量

            # ==================== 获取提示对数概率 ====================
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)  # 获取提示对数概率张量
            
            # ==================== 生成引擎核心输出 ====================
            if new_token_ids or pooler_output is not None or kv_transfer_params:  # 如果有输出数据
                # 为此请求添加EngineCoreOutput。
                outputs[request.client_index].append(  # 添加到客户端输出
                    EngineCoreOutput(
                        request_id=req_id,  # 请求ID
                        new_token_ids=new_token_ids,  # 新token ID
                        finish_reason=request.get_finished_reason(),  # 完成原因
                        new_logprobs=new_logprobs,  # 新对数概率
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,  # 新提示对数概率张量
                        pooling_output=pooler_output,  # 池化输出
                        stop_reason=request.stop_reason,  # 停止原因
                        events=request.take_events(),  # 事件
                        kv_transfer_params=kv_transfer_params,  # KV传输参数
                        trace_headers=request.trace_headers,  # 跟踪头
                        num_cached_tokens=request.num_cached_tokens,  # 缓存token数
                    )
                )
            else:  # 如果没有输出数据
                # 不变量: EngineCore不返回部分预填充输出。
                assert not prompt_logprobs_tensors  # 确保没有提示对数概率张量

        # ==================== 从运行和等待队列中移除停止的请求 ====================
        if stopped_running_reqs:  # 如果有停止的运行请求
            self.running = remove_all(self.running, stopped_running_reqs)  # 从运行队列移除
        if stopped_preempted_reqs:  # 如果有停止的被抢占请求
            # 这是一个罕见的情况，不太可能影响性能。
            self.waiting.remove_requests(stopped_preempted_reqs)  # 从等待队列移除

        # ==================== KV连接器: 更新完成的KV传输状态 ====================
        if kv_connector_output:  # 如果有KV连接器输出
            self._update_from_kv_xfer_finished(kv_connector_output)  # 更新KV传输完成状态

        # ==================== 为所有客户端创建EngineCoreOutputs ====================
        # 为此步骤中有输出的所有客户端创建EngineCoreOutputs。
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)  # 创建引擎核心输出
            for client_index, outs in outputs.items()  # 遍历客户端输出
        }

        # ==================== 处理完成的请求ID ====================
        finished_req_ids = self.finished_req_ids_dict  # 获取完成的请求ID字典
        if finished_req_ids:  # 如果有完成的请求ID
            # 包含自上次输出发送以来完成的请求ID。
            for client_index, finished_set in finished_req_ids.items():  # 遍历客户端完成集合
                # 为此客户端在EngineCoreOutputs中设置完成的请求集合。
                if (eco := engine_core_outputs.get(client_index)) is not None:  # 如果客户端有输出
                    eco.finished_requests = finished_set  # 设置完成的请求
                else:  # 如果客户端没有输出
                    engine_core_outputs[client_index] = EngineCoreOutputs(  # 创建新的输出
                        finished_requests=finished_set  # 设置完成的请求
                    )
            finished_req_ids.clear()  # 清空完成的请求ID

        # ==================== 生成统计信息 ====================
        if (
            stats := self.make_stats(spec_decoding_stats, kv_connector_stats)  # 创建统计信息
        ) is not None:  # 如果有统计信息
            # 仅向一个前端返回统计信息。
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:  # 如果没有输出
                # 即使此步骤没有请求输出，我们也必须返回统计信息。
                engine_core_outputs[0] = eco = EngineCoreOutputs()  # 创建默认输出
            eco.scheduler_stats = stats  # 设置调度器统计信息

        return engine_core_outputs  # 返回引擎核心输出

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """
        使用输出更新请求
        
        将生成的token添加到请求中并检查停止条件。
        
        Args:
            request: 请求对象
            new_token_ids: 新的token ID列表
            
        Returns:
            tuple: (修剪后的token ID列表, 是否停止)
        """
        # ==================== 添加生成的token并检查停止 ====================
        # 注意，如果请求仍在预填充中，我们期望模型运行器
        # 为此请求返回空的token ID。
        stopped = False  # 是否停止
        for num_new, output_token_id in enumerate(new_token_ids, 1):  # 遍历新token
            request.append_output_token_ids(output_token_id)  # 添加输出token ID

            # ==================== 检查停止并更新请求状态 ====================
            # 这必须在创建EngineCoreOutput之前调用。
            stopped = check_stop(request, self.max_model_len)  # 检查停止条件
            if stopped:  # 如果停止
                del new_token_ids[num_new:]  # 根据需要修剪新token
                break  # 跳出循环
        return new_token_ids, stopped  # 返回修剪后的token和停止状态

    def _free_encoder_inputs(self, request: Request) -> None:
        """
        释放编码器输入
        
        释放不再需要的编码器输入缓存。
        
        Args:
            request: 请求对象
        """
        # ==================== 获取缓存的编码器输入ID ====================
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # 优化: 如果集合为空，避免list(set)操作。
        if not cached_encoder_input_ids:  # 如果没有缓存的编码器输入
            return  # 直接返回

        # ==================== 释放编码器输入 ====================
        # 这里，我们使用list(set)来避免在迭代时修改集合。
        for input_id in list(cached_encoder_input_ids):  # 遍历缓存的编码器输入ID
            mm_feature = request.mm_features[input_id]  # 获取多模态特征
            start_pos = mm_feature.mm_position.offset  # 获取起始位置
            num_tokens = mm_feature.mm_position.length  # 获取token数
            
            # ==================== 检查是否可以释放 ====================
            if self.is_encoder_decoder and request.num_computed_tokens > 0:  # 如果是编码器-解码器且已计算token
                # 对于Whisper，一旦我们生成了单个token，
                # 我们就知道编码器输入已经完成。交叉注意力
                # KV已经被计算并缓存。
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放编码器输入
            elif start_pos + num_tokens <= request.num_computed_tokens:  # 如果编码器输出已处理
                # 编码器输出已经被处理并存储在
                # 解码器的KV缓存中。
                self.encoder_cache_manager.free_encoder_input(request, input_id)  # 释放编码器输入

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        """
        更新草稿token ID
        
        更新请求的推测解码token ID。
        
        Args:
            draft_token_ids: 草稿token ID对象
        """
        # ==================== 遍历草稿token ID ====================
        for req_id, spec_token_ids in zip(  # 遍历请求ID和推测token ID
            draft_token_ids.req_ids,  # 请求ID列表
            draft_token_ids.draft_token_ids,  # 草稿token ID列表
        ):
            # ==================== 获取请求对象 ====================
            request = self.requests.get(req_id)  # 获取请求对象
            if request is None or request.is_finished():  # 如果请求不存在或已完成
                # 请求可能已经完成。跳过。
                continue  # 继续下一个请求

            # ==================== 添加新生成的推测token ID到请求 ====================
            if not spec_token_ids:  # 如果没有推测token ID
                # 注意(woosuk): request.spec_token_ids应该被更新。
                request.spec_token_ids.clear()  # 清空推测token ID
            elif self.structured_output_manager.should_advance(request):  # 如果应该推进结构化输出
                metadata = request.structured_output_request  # 获取结构化输出请求
                request.spec_token_ids = metadata.grammar.validate_tokens(  # 验证token
                    spec_token_ids
                )
            else:  # 否则
                request.spec_token_ids = spec_token_ids  # 直接设置推测token ID

    def get_request_counts(self) -> tuple[int, int]:
        """
        获取请求计数
        
        Returns:
            tuple: (运行中的请求数, 等待中的请求数)
        """
        return len(self.running), len(self.waiting)  # 返回运行和等待请求数

    def add_request(self, request: Request) -> None:
        """
        添加请求
        
        将新请求添加到等待队列中。
        
        Args:
            request: 要添加的请求
        """
        self.waiting.add_request(request)  # 添加到等待队列
        self.requests[request.request_id] = request  # 添加到请求字典
        if self.log_stats:  # 如果记录统计信息
            request.record_event(EngineCoreEventType.QUEUED)  # 记录排队事件

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """
        完成请求
        
        处理来自调度器外部的完成信号。
        
        例如，当客户端断开连接时，API服务器可以中止请求。
        
        Args:
            request_ids: 要完成的请求ID（单个或可迭代）
            finished_status: 完成状态
        """
        assert RequestStatus.is_finished(finished_status)  # 确保是完成状态
        
        # ==================== 标准化请求ID ====================
        if isinstance(request_ids, str):  # 如果是单个字符串
            request_ids = (request_ids,)  # 转换为元组
        else:  # 否则
            request_ids = set(request_ids)  # 转换为集合

        # ==================== 初始化变量 ====================
        running_requests_to_remove = set()  # 要移除的运行请求
        waiting_requests_to_remove = []  # 要移除的等待请求
        valid_requests = []  # 有效请求

        # ==================== 第一遍: 收集要从队列中移除的请求 ====================
        for req_id in request_ids:  # 遍历请求ID
            request = self.requests.get(req_id)  # 获取请求对象
            if request is None:  # 如果请求不存在
                # 无效的请求ID。
                continue  # 继续下一个

            valid_requests.append(request)  # 添加到有效请求
            if request.status == RequestStatus.RUNNING:  # 如果是运行状态
                running_requests_to_remove.add(request)  # 添加到运行请求移除集合
            else:  # 否则
                waiting_requests_to_remove.append(request)  # 添加到等待请求移除列表

        # ==================== 从队列中移除所有请求以提高效率 ====================
        if running_requests_to_remove:  # 如果有要移除的运行请求
            self.running = remove_all(self.running, running_requests_to_remove)  # 从运行队列移除
        if waiting_requests_to_remove:  # 如果有要移除的等待请求
            self.waiting.remove_requests(waiting_requests_to_remove)  # 从等待队列移除

        # ==================== 第二遍: 设置状态并释放请求 ====================
        for request in valid_requests:  # 遍历有效请求
            request.status = finished_status  # 设置完成状态
            self._free_request(request)  # 释放请求

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        """
        释放请求
        
        释放已完成的请求的所有资源。
        
        Args:
            request: 要释放的请求
            
        Returns:
            dict[str, Any] | None: KV传输参数（如果有）
        """
        assert request.is_finished()  # 确保请求已完成

        # ==================== 处理连接器完成 ====================
        delay_free_blocks, kv_xfer_params = self._connector_finished(request)  # 获取连接器完成状态
        self.encoder_cache_manager.free(request)  # 释放编码器缓存
        request_id = request.request_id  # 获取请求ID
        self.finished_req_ids.add(request_id)  # 添加到完成请求ID集合
        if self.finished_req_ids_dict is not None:  # 如果有完成请求ID字典
            self.finished_req_ids_dict[request.client_index].add(request_id)  # 添加到客户端完成集合

        # ==================== 释放块 ====================
        if not delay_free_blocks:  # 如果不延迟释放块
            self._free_blocks(request)  # 释放块

        return kv_xfer_params  # 返回KV传输参数

    def _free_blocks(self, request: Request):
        """
        释放块
        
        释放请求的KV缓存块。
        
        Args:
            request: 要释放块的请求
        """
        assert request.is_finished()  # 确保请求已完成
        self.kv_cache_manager.free(request)  # 释放KV缓存
        del self.requests[request.request_id]  # 从请求字典中删除

    def get_num_unfinished_requests(self) -> int:
        """
        获取未完成请求数
        
        Returns:
            int: 未完成的请求总数
        """
        return len(self.waiting) + len(self.running)  # 返回等待和运行请求数之和

    def has_finished_requests(self) -> bool:
        """
        检查是否有完成的请求
        
        Returns:
            bool: 是否有完成的请求
        """
        return len(self.finished_req_ids) > 0  # 返回是否有完成的请求ID

    def reset_prefix_cache(self) -> bool:
        """
        重置前缀缓存
        
        Returns:
            bool: 是否成功重置
        """
        return self.kv_cache_manager.reset_prefix_cache()  # 重置前缀缓存

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
    ) -> SchedulerStats | None:
        """
        创建调度器统计信息
        
        Args:
            spec_decoding_stats: 推测解码统计信息
            kv_connector_stats: KV连接器统计信息
            
        Returns:
            SchedulerStats | None: 调度器统计信息（如果启用日志记录）
        """
        if not self.log_stats:  # 如果未启用日志记录
            return None  # 返回None
        
        # ==================== 创建统计信息 ====================
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()  # 获取前缀缓存统计
        assert prefix_cache_stats is not None  # 确保前缀缓存统计不为None
        return SchedulerStats(  # 返回调度器统计
            num_running_reqs=len(self.running),  # 运行请求数
            num_waiting_reqs=len(self.waiting),  # 等待请求数
            kv_cache_usage=self.kv_cache_manager.usage,  # KV缓存使用情况
            prefix_cache_stats=prefix_cache_stats,  # 前缀缓存统计
            spec_decoding_stats=spec_decoding_stats,  # 推测解码统计
            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),  # 损坏请求数
            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,  # KV连接器统计
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> SpecDecodingStats | None:
        """
        创建推测解码统计信息
        
        Args:
            spec_decoding_stats: 现有的推测解码统计信息
            num_draft_tokens: 草稿token数
            num_accepted_tokens: 接受的token数
            
        Returns:
            SpecDecodingStats | None: 推测解码统计信息（如果启用日志记录）
        """
        if not self.log_stats:  # 如果未启用日志记录
            return None  # 返回None
        
        # ==================== 创建或更新统计信息 ====================
        if spec_decoding_stats is None:  # 如果没有现有统计信息
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)  # 创建新的统计信息
        spec_decoding_stats.observe_draft(  # 观察草稿
            num_draft_tokens=num_draft_tokens,  # 草稿token数
            num_accepted_tokens=num_accepted_tokens  # 接受的token数
        )
        return spec_decoding_stats  # 返回统计信息

    def shutdown(self) -> None:
        """
        关闭调度器
        
        清理所有资源并关闭相关组件。
        """
        # ==================== 关闭KV事件发布器 ====================
        if self.kv_event_publisher:  # 如果有KV事件发布器
            self.kv_event_publisher.shutdown()  # 关闭发布器
        
        # ==================== 关闭连接器 ====================
        if self.connector is not None:  # 如果有连接器
            self.connector.shutdown()  # 关闭连接器

    ########################################################################
    # KV连接器相关方法
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        """
        获取KV连接器
        
        Returns:
            KVConnectorBase_V1 | None: KV连接器实例（如果有）
        """
        return self.connector  # 返回连接器

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        调用KV连接器的request_finished()方法（如果适用）
        
        返回可选的KV传输参数，这些参数将包含在请求输出中。
        
        Args:
            request: 已完成的请求
            
        Returns:
            tuple: (是否延迟释放块, KV传输参数)
        """
        if self.connector is None:  # 如果没有连接器
            return False, None  # 返回不延迟释放和None

        # ==================== 获取块ID并调用连接器 ====================
        (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)  # 获取块ID
        return self.connector.request_finished(request, block_ids)  # 调用连接器完成方法

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV连接器: 检查request_id是否已完成接收
        
        finished_recving_kv_req_ids列表是在前一步的update_from_output中
        基于工作器端连接器填充的。
        
        当KV传输准备就绪时，我们缓存块，
        请求状态将从WAITING_FOR_REMOTE_KV移回WAITING。
        
        Args:
            request: 等待远程KV的请求
            
        Returns:
            bool: 是否准备就绪
        """
        assert self.connector is not None  # 确保连接器存在
        if request.request_id not in self.finished_recving_kv_req_ids:  # 如果请求不在完成接收列表中
            return False  # 返回未准备就绪

        # ==================== 处理KV加载失败 ====================
        if request.request_id in self.failed_recving_kv_req_ids:  # 如果请求在失败接收列表中
            # 请求有KV加载失败；num_computed_tokens已经在
            # _update_requests_with_invalid_blocks中更新
            if request.num_computed_tokens:  # 如果有已计算的token
                # 缓存任何有效的已计算token。
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)  # 缓存块
            else:  # 如果没有有效的已计算token
                # 没有有效的已计算token，释放分配的块。
                # 重试时可能有本地缓存命中。
                self.kv_cache_manager.free(request)  # 释放请求

            self.failed_recving_kv_req_ids.remove(request.request_id)  # 从失败列表移除
        else:  # 如果请求成功接收
            # ==================== 缓存块 ====================
            # 现在块已准备就绪，实际缓存它们。
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)  # 获取块ID
            num_computed_tokens = len(block_ids) * self.block_size  # 计算已计算token数
            # 处理请求token数少于一个块的情况。
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)  # 限制在请求token数内
            if num_computed_tokens == request.num_tokens:  # 如果等于请求token数
                num_computed_tokens -= 1  # 减1
            # 这将缓存块（如果启用了缓存）。
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)  # 缓存块

            # ==================== 更新请求状态 ====================
            # 更新请求状态以便调度。
            request.num_computed_tokens = num_computed_tokens  # 设置已计算token数

        # ==================== 返回准备就绪状态 ====================
        # 返回我们已准备就绪。
        self.finished_recving_kv_req_ids.remove(request.request_id)  # 从完成接收列表移除
        return True  # 返回准备就绪

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV连接器: 根据输出更新调度器状态
        
        工作器端连接器将finished_recving和finished_sending请求添加到输出中。
        * 如果finished_sending: 释放块
        * 如果finished_recving: 添加到状态，以便我们可以在下一步调度请求。
        
        Args:
            kv_connector_output: KV连接器输出
        """

        # ==================== 更新连接器输出 ====================
        if self.connector is not None:  # 如果有连接器
            self.connector.update_connector_output(kv_connector_output)  # 更新连接器输出

        # ==================== 更新接收和发送状态 ====================
        # KV连接器: 从上一步更新接收和发送状态。
        for req_id in kv_connector_output.finished_recving or ():  # 遍历完成接收的请求
            logger.debug("Finished recving KV transfer for request %s", req_id)  # 记录调试信息
            self.finished_recving_kv_req_ids.add(req_id)  # 添加到完成接收集合
        for req_id in kv_connector_output.finished_sending or ():  # 遍历完成发送的请求
            logger.debug("Finished sending KV transfer for request %s", req_id)  # 记录调试信息
            if req_id not in self.requests:  # 如果请求不在请求字典中
                logger.warning(
                    "Got finished sending KV transfer for request %s,"
                    "but the request is already freed.",
                    req_id,  # 记录警告信息
                )
            else:  # 如果请求存在
                self._free_blocks(self.requests[req_id])  # 释放请求的块

    def _update_requests_with_invalid_blocks(
        self, requests: Iterable[Request], invalid_block_ids: set[int]
    ) -> tuple[set[str], int]:
        """
        识别并更新受无效KV缓存块影响的请求
        
        此方法扫描给定的请求，检测那些有无效块的请求，
        并将它们的`num_computed_tokens`调整到最长的有效前缀。
        为了可观察性，它还累积所有受影响请求中需要重新计算的token总数。
        
        Args:
            requests: 要扫描无效块的请求集合
            invalid_block_ids: 无效块的ID
            
        Returns:
            tuple:
                - affected_req_ids (set[str]): 受无效块影响的请求ID
                - total_affected_tokens (int): 所有受影响请求中必须重新计算的token总数（用于可观察性）
        """
        # ==================== 初始化变量 ====================
        affected_req_ids: set[str] = set()  # 受影响的请求ID
        total_affected_tokens = 0  # 总受影响token数
        # 如果一个块无效且被批次中的多个请求共享，
        # 这些请求必须重新调度，但只有第一个会重新计算它。
        # 此集合跟踪已标记为重新计算的块。
        marked_invalid_block_ids: set[int] = set()  # 已标记的无效块ID
        
        # ==================== 遍历请求 ====================
        for request in requests:  # 遍历请求
            is_affected = False  # 是否受影响
            marked_invalid_block = False  # 是否标记了无效块
            req_id = request.request_id  # 获取请求ID
            # TODO (davidb): 添加混合内存分配器支持
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)  # 获取请求块ID
            
            # ==================== 确定已计算token数 ====================
            # 我们只迭代可能包含外部计算token的块
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 如果状态是等待远程KV
                # 异步加载。如果设置了num_computed_tokens，这意味着我们
                # 已经在之前的步骤中处理了一些块失败
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids  # 如果在失败接收列表中
                    else len(req_block_ids) * self.block_size  # 否则使用块数乘以块大小
                )
            else:  # 否则
                # 同步加载。num_computed_tokens包括新token
                req_num_computed_tokens = request.num_cached_tokens  # 使用缓存token数

            # ==================== 计算已计算块数 ====================
            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1  # 计算已计算块数
            ) // self.block_size
            
            # ==================== 检查无效块 ====================
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):  # 遍历已计算块
                if block_id not in invalid_block_ids:  # 如果块不在无效块ID中
                    continue  # 继续下一个块

                is_affected = True  # 标记为受影响

                if block_id in marked_invalid_block_ids:  # 如果块已标记为无效
                    # 此无效块与之前的请求共享
                    # 并已被标记为重新计算。
                    # 这意味着此请求在重新调度时仍可以将此块
                    # 视为已计算。
                    # 目前这仅适用于同步加载；异步
                    # 加载尚不支持块共享
                    continue  # 继续下一个块

                marked_invalid_block_ids.add(block_id)  # 添加到已标记集合

                if marked_invalid_block:  # 如果已标记了无效块
                    # 此请求已经为重新计算标记了一个无效块
                    # 并更新了其num_computed_tokens。
                    continue  # 继续下一个块

                marked_invalid_block = True  # 标记为已标记无效块
                # ==================== 截断已计算token ====================
                # 在第一个失败块处截断已计算的token
                request.num_computed_tokens = idx * self.block_size  # 设置已计算token数
                total_affected_tokens += (
                    req_num_computed_tokens - request.num_computed_tokens  # 增加受影响token数
                )

            # ==================== 处理受影响的请求 ====================
            if is_affected:  # 如果请求受影响
                if not marked_invalid_block:  # 如果没有标记无效块
                    # 此请求的所有无效块都与
                    # 之前的请求共享，并将由它们重新计算。
                    # 恢复到仅将缓存的token视为已计算。
                    # 目前这仅适用于同步加载；异步
                    # 加载尚不支持块共享
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens  # 增加受影响token数
                    )
                    request.num_computed_tokens = request.num_cached_tokens  # 设置为缓存token数

                affected_req_ids.add(request.request_id)  # 添加到受影响请求ID

        return affected_req_ids, total_affected_tokens  # 返回受影响请求ID和总受影响token数

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        处理无效块
        
        处理KV加载失败，识别受影响的请求并调整它们的计算状态。
        
        Args:
            invalid_block_ids: 无效块的ID集合
            
        Returns:
            set[str]: 受影响的运行请求ID（在update_from_output中跳过）
        """
        # ==================== 初始化统计 ====================
        total_requests_to_reschedule = 0  # 要重新调度的总请求数
        total_tokens_to_reschedule = 0  # 要重新调度的总token数

        # ==================== 处理异步KV加载 (WAITING_FOR_REMOTE_KVS) ====================
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS  # 等待远程KV的请求
        )
        async_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(  # 更新异步加载请求
                async_load_reqs, invalid_block_ids
            )
        )

        total_requests_to_reschedule += len(async_affected_req_ids)  # 增加异步请求数
        total_tokens_to_reschedule += num_tokens_to_reschedule  # 增加异步token数

        # ==================== 标记异步KV加载失败的请求 ====================
        # 标记有异步KV加载失败的请求；它们将在加载完成后重新调度
        self.failed_recving_kv_req_ids |= async_affected_req_ids  # 添加到失败接收集合

        # ==================== 处理同步KV加载 (运行中的请求) ====================
        sync_affected_req_ids, num_tokens_to_reschedule = (
            self._update_requests_with_invalid_blocks(self.running, invalid_block_ids)  # 更新同步加载请求
        )

        total_requests_to_reschedule += len(sync_affected_req_ids)  # 增加同步请求数
        total_tokens_to_reschedule += num_tokens_to_reschedule  # 增加同步token数

        # ==================== 记录恢复信息 ====================
        if total_requests_to_reschedule:  # 如果有要重新调度的请求
            logger.warning(
                "Recovered from KV load failure: "
                "%d request(s) rescheduled (%d tokens affected).",
                total_requests_to_reschedule,  # 记录重新调度请求数
                total_tokens_to_reschedule,  # 记录受影响token数
            )

        # ==================== 返回受影响的运行请求ID ====================
        # 返回受影响的运行请求ID，以便在update_from_output中跳过。
        return sync_affected_req_ids  # 返回同步受影响的请求ID
