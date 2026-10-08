///|
/// The pinned split-KV device function also supports a single partition.
/// No append, dropout, RNG, LSE output or host framework is used.
fn flashattention_source(plan : ReferenceAttentionPlan) -> String {
  let (heads, kv_heads, dimension, page) = plan.abi.shape()
  let (width, stride, _) = plan.abi.layout()
  let (rows, tokens, _) = plan.abi.limits()
  let prefill = plan.backend != FlashAttentionDecode
  let mixed = plan.backend == FlashAttentionMixed
  let query_tile = 64
  let group = heads / kv_heads
  let source =
    $|#include <optional>
    $|#include "flash.h"
    $|#include "flash_fwd_kernel.h"
    $|#ifndef LUNAFLUX_FLASH_CSR_TAIL_SAFE_V1
    $|#error "The pinned FlashAttention headers must use the bounded CSR tail resolver"
    $|#endif
    $|using LfReferenceTraits = Flash_fwd_kernel_traits<128,64,128,4,false,false,cutlass::bfloat16_t>;
    $|static_assert(LfReferenceTraits::kSmemSize == 81920);
    $|extern "C" __global__ void \{plan.symbol}(
    $| const int *counts, const int *positions, const int *offsets,
    $| const int *lengths, const int *page_offsets, const int *page_indices,
    $| const __nv_bfloat16 *input, __nv_bfloat16 *output,
    $| const __nv_bfloat16 *keys, const __nv_bfloat16 *values) {
    $| if (blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
    $|     counts[0] < 0 || counts[0] > counts[2] ||
    $|     counts[2] <= 0 || counts[2] > \{rows} || counts[3] <= 0 || counts[3] > \{tokens}) return;
    $| \{if mixed { "if (counts[0] == 0) return;" } else { "" }}
    $| int tile = blockIdx.x;
    $| for (;;) {
    $|   int row = 0, local = tile;
    $|   for (; row < counts[2]; ++row) {
    $|     const int n = offsets[row+1]-offsets[row];
    $|     const bool eligible = \{if mixed { "true" } else if prefill { "row < counts[0]" } else { "row >= counts[0] && n == 1" }};
    $|     const int tiles = eligible ? (n+\{query_tile - 1})/\{query_tile} : 0;
    $|     if (local < tiles) break;
    $|     local -= tiles;
    $|   }
    $|   if (row == counts[2]) return;
    $|   const int begin = offsets[row], n = offsets[row+1]-begin;
    $|   flash::Flash_fwd_params p{};
    $|   // Decode reinterprets adjacent grouped heads as query rows, sharing
    $|   // one K/V tile without changing their positions or value semantics.
    $|   p.q_ptr = const_cast<__nv_bfloat16 *>(input + (unsigned long long)begin*\{width}\{if prefill { "" } else { " + blockIdx.y*" + (group * dimension).to_string() }});
    $|   p.k_ptr = const_cast<__nv_bfloat16 *>(keys\{if prefill { "" } else { " + blockIdx.y*" + dimension.to_string() }});
    $|   p.v_ptr = const_cast<__nv_bfloat16 *>(values\{if prefill { "" } else { " + blockIdx.y*" + dimension.to_string() }});
    $|   p.o_ptr = output + (unsigned long long)begin*\{heads * dimension}\{if prefill { "" } else { " + blockIdx.y*" + (group * dimension).to_string() }};
    $|   p.q_row_stride=\{if prefill { width } else { dimension }}; p.q_head_stride=\{dimension};
    $|   p.k_batch_stride=\{stride}; p.v_batch_stride=\{stride};
    $|   p.k_row_stride=\{kv_heads * dimension}; p.v_row_stride=\{kv_heads * dimension};
    $|   p.k_head_stride=\{dimension}; p.v_head_stride=\{dimension};
    $|   p.o_row_stride=\{if prefill { heads * dimension } else { dimension }}; p.o_head_stride=\{dimension};
    $|   p.h=\{if prefill { heads } else { 1 }}; p.h_k=\{if prefill { kv_heads } else { 1 }}; p.h_h_k_ratio=\{if prefill { group } else { 1 }};
    $|   p.b=1; p.seqlen_q=\{if prefill { "n" } else { group.to_string() }}; p.seqlen_k=lengths[row]; p.d=\{dimension};
    $|   p.d_rounded=\{dimension}; p.seqlen_q_rounded=(n+63)/64*64;
    $|   p.seqlen_k_rounded=(lengths[row]+127)/128*128; p.total_q=n;
    $|   p.block_table=const_cast<int *>(page_indices + page_offsets[row]); p.page_block_size=\{page};
    $|   p.scale_softmax=0.08838834764831845f; p.scale_softmax_log2=p.scale_softmax*1.4426950408889634f;
    $|   p.p_dropout=1.0f; p.rp_dropout=1.0f; p.scale_softmax_rp_dropout=p.scale_softmax;
    $|   p.is_bf16=true; p.is_causal=\{if prefill { "true" } else { "false" }}; p.window_size_left=-1; p.window_size_right=0;
    $|   p.num_splits=1;
    $|   // Chunk queries are the contiguous causal suffix of this sequence.
    $|   if (positions[begin] != lengths[row]-n || positions[begin+n-1] != lengths[row]-1) return;
    $|   flash::compute_attn_1rowblock_splitkv<LfReferenceTraits,\{if prefill { "true" } else { "false" }},false,false,false,true,false,false,false>(p,0,\{if prefill { "blockIdx.y" } else { "0" }},local,0,1);
    $|   __syncthreads();
    $|   tile += gridDim.x;
    $| }
    $|}
  source
}
