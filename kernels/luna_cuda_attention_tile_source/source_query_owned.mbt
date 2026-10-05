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
  let (heads, _, head, _) = abi.shape()
  let (row_width, _, _) = abi.layout()
  let (max_rows, max_tokens, max_entries) = abi.limits()
  let partitions = lowering.key_value_partitions()
  let asynchronous = lowering.copy() == CudaAsynchronousSharedLoad
  let stages = lowering.pipeline().0
  if !valid_cuda_symbol(symbol) ||
    maximum_source_bytes <= 0 ||
    maximum_source_bytes > 16777216 ||
    !lowering.query_owned_fragments() ||
    lowering.compute() != CudaMatrixQkPvMultiplyAccumulate ||
    (!asynchronous && lowering.copy() != CudaCooperativeSharedLoad) ||
    lowering.pipeline().1 != 16 ||
    (stages != 1 && stages != 2) ||
    (!asynchronous && stages != 1) ||
    head != d ||
    threads != q / 16 * 32 ||
    partitions <= 0 ||
    shared < (q + 2 * k * stages).to_int64() * d.to_int64() * 2L + 16L {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let layout = cuda_attention_operand_layout(q, k, d)
  guard cuda_physical_program(lowering) is QueryFragmentProgram(fold, effects) &&
    fold.geometry() == (q, k, d) &&
    lowering.prefill_fragments() is Some(fragments) &&
    fragments.program().fold() == fold else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let split = effects.split_key_value()
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
    $|\{cuda_attention_matrix_fragments(shared_scores=false, forwarded_right=fragments)}
    $|\{cuda_query_owned_score_fold()}
    $|\{cuda_query_owned_numeric_primitives(law=fold.exponential_law())}
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
    $|  auto *vbuf = kbuf + \{stages} * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION;
    $|  auto *control = reinterpret_cast<int *>(vbuf + \{stages} * LF_KEY_VALUE_TILE * LF_HEAD_DIMENSION);
    $|  for (int tile = blockIdx.x; ; tile += gridDim.x) {
    $|\{cuda_query_tile_binding(abi, q)}
    $|    for (int vector=thread; vector<LF_QUERY_TILE*LF_HEAD_DIMENSION/8; vector+=LF_BLOCK_THREADS) {
    $|      const int r = vector*8/LF_HEAD_DIMENSION, c = vector*8%LF_HEAD_DIMENSION;
    $|      uint4 packed = make_uint4(0,0,0,0);
    $|      if (r<active) packed = *reinterpret_cast<const uint4 *>(input+(unsigned long long)(token_base+r)*\{row_width}ULL+head*LF_HEAD_DIMENSION+c);
    $|      *reinterpret_cast<uint4 *>(qbuf+lf_operand_index(r,c,LF_HEAD_DIMENSION)) = packed;
    $|    }
    $|    \{cuda_query_owned_effects(effects, PublishQueryOperands)}
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
    $|\{cuda_query_operand_bootstrap(fragments)}\{cuda_query_owned_stage(abi, lowering.reads_dense_current_key_value(), asynchronous, split~, affine_positions=lowering.affine_current_positions(), key_rows=k)}
    $|    int slot=0;
    $|    if (key_begin<key_end) { \{cuda_query_owned_effects(effects, PrimeQueryFold)} }
    $|    for (int kb=key_begin; kb<key_end; kb+=LF_KEY_VALUE_TILE) {
    $|      \{cuda_query_owned_effects(effects, IssueCurrentKeyValue)}
    $|      \{cuda_query_owned_effects(effects, AwaitKeyReadiness)}
    $|      \{cuda_query_owned_effects(effects, PublishKeyReadiness)}
    $|      if (!control[0]) { \{cuda_query_owned_effects(effects, InvalidQueryTileExit)} return; }
    $|      const auto *current_k=kbuf+slot*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION;
    $|      const auto *current_v=vbuf+slot*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION;
    $|      \{cuda_query_owned_effects(effects, AfterKeyPublication)}
    $|      \{if split { "LfMatrixAccumulator score[LF_KEY_VALUE_TILE/16] = {};" } else { "" }}
    $|      if (qr<active) {
    $|        \{if split { "" } else { "LfMatrixAccumulator score[LF_KEY_VALUE_TILE/16] = {};" }}
    $|\{cuda_online_fold_phase(fold.score_phase(), fragments)}
    $|      \{if split { "}" } else { "" }}
    $|      \{cuda_query_owned_effects(effects, BeforeValueComputation)}
    $|      \{if split { "if (qr<active) {" } else { "" }}
    $|\{cuda_online_fold_phase(fold.value_phase(), fragments)}
    $|      }
    $|      \{cuda_query_owned_effects(effects, ReleaseValueReaders)}
    $|      \{cuda_query_owned_effects(effects, AfterValueRelease)}
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
    $|    \{cuda_query_owned_effects(effects, ReleaseQueryReaders)}
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
