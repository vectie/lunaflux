///|
/// Realizes the query-owned map/fold with a single register layout spanning
/// QK, probability conversion and PV. Only Q/K/V operands use shared memory.
/// The four-lane row reduction has its own schedule numerical identity.
fn emit_cuda_query_owned_attention_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  symbol : String,
  maximum_source_bytes : Int,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let (q, k, d) = lowering.tile()
  let (_, _, threads, shared) = lowering.launch()
  let (heads, _, head, page_tokens) = abi.shape()
  let (row_width, _, _) = abi.layout()
  let (max_rows, max_tokens, max_entries) = abi.limits()
  let partitions = lowering.key_value_partitions()
  if !valid_cuda_symbol(symbol) ||
    maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 ||
    !lowering.query_owned_fragments() ||
    lowering.compute() != CudaMatrixQkPvMultiplyAccumulate ||
    lowering.copy() != CudaCooperativeSharedLoad ||
    lowering.pipeline() != (1, 16) ||
    head != d ||
    threads != q / 16 * 32 ||
    partitions <= 0 ||
    shared < (q + 2 * k).to_int64() * d.to_int64() * 2L + 16L {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let layout = cuda_attention_operand_layout(q, k, d)
  let output_parameters = if partitions == 1 {
    "const __nv_bfloat16 *input, __nv_bfloat16 *output, const __nv_bfloat16 *keys, const __nv_bfloat16 *values"
  } else {
    "const __nv_bfloat16 *input, const __nv_bfloat16 *keys, const __nv_bfloat16 *values, float *workspace"
  }
  let states = max_tokens.to_int64() * heads.to_int64() * partitions.to_int64()
  let terminal = if partitions == 1 {
    "if (query<active) output[((unsigned long long)(token_base+query)*\{heads}ULL+head)*LF_HEAD_DIMENSION+component]=__float2bfloat16_rn(out[column].x[item]/denominator[item%4/2]);"
  } else {
    "if (query<active) {\n" +
    "            const unsigned long long state=((unsigned long long)(token_base+query)*\{heads}ULL+head)*\{partitions}ULL+blockIdx.z;\n" +
    "            workspace[2ULL*\{states}ULL+state*LF_HEAD_DIMENSION+component]=out[column].x[item];\n" +
    "            if (component==0) { workspace[state]=maximum[item%4/2]; workspace[\{states}ULL+state]=denominator[item%4/2]; }\n" +
    "          }"
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|#define LF_QUERY_TILE \{q}
    $|#define LF_KEY_VALUE_TILE \{k}
    $|#define LF_HEAD_DIMENSION \{d}
    $|#define LF_BLOCK_THREADS \{threads}
    $|#define LF_SHARED_MEMORY_BYTES \{shared}
    $|#define LF_KEY_VALUE_PARTITIONS \{partitions}
    $|#define LF_PARTIAL_STATE_COUNT \{states}ULL
    $|// query-owner-v1: ordered QK/PV, BF16 probabilities, four-owner softmax.
    $|\{layout}
    $|\{cuda_attention_matrix_fragments(shared_scores=false)}
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *positions, const int *offsets,
    $|    const int *lengths, const int *page_offsets, const int *page_indices,
    $|    \{output_parameters}) {
    $|  const int thread = threadIdx.x, lane = thread & 31, warp = thread / 32;
    $|  const int head = blockIdx.y, rows = counts[2], tokens = counts[3];
    $|  if (blockDim.x != LF_BLOCK_THREADS || blockIdx.z >= \{partitions} || head >= \{heads} || rows <= 0 ||
    $|      rows > \{max_rows} || tokens <= 0 || tokens > \{max_tokens} ||
    $|      counts[0] < 0 || counts[1] < 0 || counts[0] + counts[1] != rows ||
    $|      counts[4] <= 0 || counts[4] > \{max_entries}) return;
    $|  extern __shared__ __align__(16) unsigned char arena[];
    $|  auto *qbuf = reinterpret_cast<__nv_bfloat16 *>(arena);
    $|  auto *kbuf = qbuf + LF_QUERY_TILE * LF_HEAD_DIMENSION;
    $|  auto *vbuf = kbuf + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  auto *control = reinterpret_cast<int *>(vbuf + LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
    $|  for (int tile = blockIdx.x; ; tile += gridDim.x) {
    $|    int row = -1, token_base = 0, prefix = 0;
    $|    for (int r = 0; r < rows; ++r) {
    $|      const int begin = offsets[r], end = offsets[r+1];
    $|      const int tiles = end > begin ? (end-begin+LF_QUERY_TILE-1)/LF_QUERY_TILE : 0;
    $|      if (tile < prefix + tiles) { row = r; token_base = begin + (tile-prefix)*LF_QUERY_TILE; break; }
    $|      prefix += tiles;
    $|    }
    $|    if (row < 0) return;
    $|    const int begin = offsets[row], end = offsets[row+1], length = lengths[row];
    $|    const int page_begin = page_offsets[row], page_end = page_offsets[row+1];
    $|    if (begin < 0 || end <= begin || end > tokens || length <= 0 ||
    $|        page_begin < 0 || page_end <= page_begin || page_end > counts[4] ||
    $|        page_end-page_begin != (length+\{page_tokens-1})/\{page_tokens}) return;
    $|    const int active = min(LF_QUERY_TILE, end-token_base);
    $|    if (thread == 0) {
    $|      control[0] = 1; control[1] = 0;
    $|      for (int i=0; i<active; ++i) {
    $|        const int position = positions[token_base+i];
    $|        if (position < 0 || position >= length) control[0] = 0;
    $|        control[1] = max(control[1], position+1);
    $|      }
    $|    }
    $|    for (int vector=thread; vector<LF_QUERY_TILE*LF_HEAD_DIMENSION/8; vector+=LF_BLOCK_THREADS) {
    $|      const int r = vector*8/LF_HEAD_DIMENSION, c = vector*8%LF_HEAD_DIMENSION;
    $|      uint4 packed = make_uint4(0,0,0,0);
    $|      if (r<active) packed = *reinterpret_cast<const uint4 *>(input+(unsigned long long)(token_base+r)*\{row_width}ULL+head*LF_HEAD_DIMENSION+c);
    $|      *reinterpret_cast<uint4 *>(qbuf+lf_operand_index(r,c,LF_HEAD_DIMENSION)) = packed;
    $|    }
    $|    __syncthreads();
    $|    if (!control[0]) return;
    $|    const int total_keys = control[1], qr = warp*16;
    $|    const int each = total_keys/\{partitions}, larger = total_keys%\{partitions}, split = blockIdx.z;
    $|    const int key_begin = split*each+min(split,larger);
    $|    const int key_end = key_begin+each+(split<larger?1:0);
    $|    const int query0 = qr+lane/4, query1 = query0+8;
    $|    const int p0 = query0<active ? positions[token_base+query0] : -1;
    $|    const int p1 = query1<active ? positions[token_base+query1] : -1;
    $|    const float scale = rsqrtf((float)LF_HEAD_DIMENSION);
    $|    float maximum[2] = {-CUDART_INF_F,-CUDART_INF_F}, denominator[2] = {};
    $|    LfMatrixAccumulator out[LF_HEAD_DIMENSION/16] = {};
    $|    for (int kb=key_begin; kb<key_end; kb+=LF_KEY_VALUE_TILE) {
    $|      for (int vector=thread; vector<LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8; vector+=LF_BLOCK_THREADS) {
    $|        const int r = vector*8/LF_HEAD_DIMENSION, c = vector*8%LF_HEAD_DIMENSION;
    $|        const int position = kb+r;
    $|\{cuda_prefill_key_value_view_address(abi, true, dense=lowering.reads_dense_current_key_value())}
    $|        uint4 pk=make_uint4(0,0,0,0), pv=pk;
    $|        if (kv_source_valid) {
    $|          pk=*reinterpret_cast<const uint4 *>(kv_key_source);
    $|          pv=*reinterpret_cast<const uint4 *>(kv_value_source);
    $|        }
    $|        *reinterpret_cast<uint4 *>(kbuf+lf_operand_index(r,c,LF_HEAD_DIMENSION))=pk;
    $|        *reinterpret_cast<uint4 *>(vbuf+lf_operand_index(r,c,LF_HEAD_DIMENSION))=pv;
    $|      }
    $|      __syncthreads();
    $|      if (!control[0]) return;
    $|      if (qr<active) {
    $|        LfMatrixAccumulator score[LF_KEY_VALUE_TILE/16] = {};
    $|        #pragma unroll
    $|        for (int reduction=0; reduction<LF_HEAD_DIMENSION; reduction+=16) {
    $|          const auto a=lf_matrix_load(qbuf,qr,reduction,LF_HEAD_DIMENSION);
    $|          #pragma unroll
    $|          for (int key=0; key<LF_KEY_VALUE_TILE/16; ++key) {
    $|            const auto b=lf_matrix_load(kbuf,key*16,reduction,LF_HEAD_DIMENSION);
    $|            lf_matrix_mma(score[key],a,b);
    $|          }
    $|        }
    $|        float next[2]={-CUDART_INF_F,-CUDART_INF_F};
    $|        #pragma unroll
    $|        for (int key=0; key<LF_KEY_VALUE_TILE/16; ++key) {
    $|          #pragma unroll
    $|          for (int item=0; item<8; ++item) {
    $|            const int r=item%4/2, position=kb+key*16+lf_matrix_column(item,lane);
    $|            const float x=position<key_end && position<=(r==0?p0:p1) ? score[key].x[item]*scale : -CUDART_INF_F;
    $|            score[key].x[item]=x;
    $|            next[r]=fmaxf(next[r],x);
    $|          }
    $|        }
    $|        float alpha[2], sum[2]={};
    $|        #pragma unroll
    $|        for (int r=0;r<2;++r) {
    $|          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffU,next[r],1,4));
    $|          next[r]=fmaxf(next[r],__shfl_xor_sync(0xffffffffU,next[r],2,4));
    $|          next[r]=fmaxf(next[r],maximum[r]);
    $|          alpha[r]=isinf(maximum[r])?0.0f:expf(maximum[r]-next[r]);
    $|          maximum[r]=next[r];
    $|        }
    $|        #pragma unroll
    $|        for (int column=0;column<LF_HEAD_DIMENSION/16;++column) {
    $|          #pragma unroll
    $|          for (int item=0;item<8;++item) out[column].x[item]*=alpha[item%4/2];
    $|        }
    $|        #pragma unroll
    $|        for (int key=0;key<LF_KEY_VALUE_TILE/16;++key) {
    $|          float weight[8];
    $|          #pragma unroll
    $|          for (int item=0;item<8;++item) {
    $|            const int r=item%4/2;
    $|            weight[item]=isinf(score[key].x[item])?0.0f:expf(score[key].x[item]-next[r]);
    $|            sum[r]+=weight[item];
    $|          }
    $|          LfMatrixOperand probability;
    $|          #pragma unroll
    $|          for (int pair=0;pair<4;++pair) {
    $|            const unsigned lo=__bfloat16_as_ushort(__float2bfloat16_rn(weight[pair*2]));
    $|            const unsigned hi=__bfloat16_as_ushort(__float2bfloat16_rn(weight[pair*2+1]));
    $|            probability.x[pair]=lo|(hi<<16);
    $|          }
    $|          #pragma unroll
    $|          for (int column=0;column<LF_HEAD_DIMENSION/16;++column) {
    $|            const auto value=lf_matrix_load(vbuf,key*16,column*16,LF_HEAD_DIMENSION,true);
    $|            lf_matrix_mma(out[column],probability,value,true);
    $|          }
    $|        }
    $|        #pragma unroll
    $|        for (int r=0;r<2;++r) {
    $|          sum[r]+=__shfl_xor_sync(0xffffffffU,sum[r],1,4);
    $|          sum[r]+=__shfl_xor_sync(0xffffffffU,sum[r],2,4);
    $|          denominator[r]=denominator[r]*alpha[r]+sum[r];
    $|        }
    $|      }
    $|      __syncthreads();
    $|    }
    $|    if (qr<active) {
    $|      #pragma unroll
    $|      for (int column=0;column<LF_HEAD_DIMENSION/16;++column) {
    $|        #pragma unroll
    $|        for (int item=0;item<8;++item) {
    $|          const int query=qr+lf_matrix_row(item,lane), component=column*16+lf_matrix_column(item,lane);
    $|          \{terminal}
    $|        }
    $|      }
    $|    }
    $|    __syncthreads();
    $|  }
    $|}
  let bytes = @utf8.encode(source)
  if bytes.length() > maximum_source_bytes {
    raise InvalidCudaAttentionTileSource(SourceLimit)
  }
  {
    lowering_digest: lowering.digest(),
    abi,
    symbol,
    bytes,
    digest: CudaAttentionTileSourceDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    ),
  }
}
