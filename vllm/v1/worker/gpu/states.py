# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


class RequestState:
    """V2 ModelRunner 的常驻请求状态表（GPU / UVA 显存驻留）。

    核心设计思想：
    - 预分配容量为 max_num_reqs 的固定槽位（通过 req_id_to_index / free_indices 管理槽位分配与回收）。
    - 新请求加入时通过 StagedWriteTensor / UvaBackedTensor 暂存 CPU 写入，并在本步统一提交（apply_staged_writes）；
      其中超大的 all_token_ids 默认放在 UVA（GPU 可零拷贝直读的锁页主机内存）以节省数 GB 显存。
    - Decode 阶段由 GPU Triton kernel（post_update）直接在显存中原地更新 num_computed_tokens 与
      last_sampled_tokens，彻底消除了传统架构每步在 CPU 重建全量请求输入与 H2D 拷贝的开销。

    Sample（假设 max_num_reqs=256, max_model_len=8192, num_speculative_steps=2，当前有 3 个活跃请求）：
        req_id_to_index = {"req_A": 1, "req_B": 4, "req_C": 9}  # 请求固定槽位号（整个生命周期不变）

        # 各状态数组在槽位 1, 4, 9 上的快照：
        # - 槽位 1 (req_A, 常规 Decode): prompt 20 tok，已算完 21 tok，上一步刚采样出 token 888
        # - 槽位 4 (req_B, 投机 Decode): prompt 10 tok，已算完 10 tok，上一步采样出 501，草稿器预测了 [502, 503]
        # - 槽位 9 (req_C, 新入队 Chunked Prefill): prompt 40 tok，命中前缀缓存 16 tok，尚未产出过采样 token
        prompt_len.gpu          -> slot[1]=20,  slot[4]=10,         slot[9]=40
        prefill_len.gpu         -> slot[1]=20,  slot[4]=10,         slot[9]=40          # 若抢占重算含已生成token则 > prompt_len
        num_computed_tokens.gpu -> slot[1]=21,  slot[4]=10,         slot[9]=16
        last_sampled_tokens     -> slot[1]=[888], slot[4]=[501],    slot[9]=[0]         # shape: [256, 1]
        draft_tokens            -> slot[1]=[0,0], slot[4]=[502,503], slot[9]=[0,0]      # shape: [256, 2]
        all_token_ids.gpu       -> shape [256, 8192] (UVA 零拷贝内存):
                                   slot[9, :40] = [11, 22, ..., 101, 102, ..., 140]
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        num_prefill_lookahead: int = 1,
        use_dense_all_token_ids: bool = False,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU by default, but
        # ngram_gpu benefits from dense device residency because it scans
        # active rows repeatedly during proposal.
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=not use_dense_all_token_ids,
        )
        # NOTE(woosuk): Distinguish clearly between prompt_len and prefill_len:
        # - prompt_len: Number of tokens in the user-provided prompt.
        # - prefill_len: Number of tokens passed into the model runner.
        #   This can include the prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len. It grows as the request progresses.
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Max total seq length (prompt_len + max_tokens).
        self.max_seq_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead, self.max_num_reqs, dtype=torch.int32, device=device
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

        if self.num_speculative_steps > 0:
            self.draft_tokens[req_idx].zero_()

    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> int | None:
        """Return the freed slot index, or None if the request was not found."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
        return req_idx
