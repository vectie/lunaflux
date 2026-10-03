///|
/// Matrix rows are GQA heads of one query, not fabricated temporal rows. The
/// same ordered fragment fold as prefill is reused under an explicit BF16
/// probability law; inactive rows execute collectives but never write output.
fn emit_cuda_grouped_matrix_decode_source(
  lowering : @luna_cuda_attention_tile_lowering.CudaAttentionTileLowering,
  abi : CudaPagedAttentionTileAbi,
  requested_symbol : String,
  maximum_source_bytes : Int,
  matrix : @attention_physical_ir.GroupedMatrixDecodeFold,
  effects : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
) -> CudaAttentionTileSource raise CudaAttentionTileSourceError {
  let (q, k, d) = lowering.tile()
  let (_, grid_y, threads, shared) = lowering.launch()
  let (heads, kv_heads, head, page_rows) = abi.shape()
  let (input_width, _, _) = abi.layout()
  let (max_rows, max_tokens, max_pages) = abi.limits()
  let group = heads / kv_heads
  let stages = if effects.is_pipelined() { 2 } else { 1 }
  guard !abi.query_metadata &&
    lowering.phase() == Decode &&
    q == 1 &&
    lowering.compute() == CudaMatrixQkPvMultiplyAccumulate &&
    head == d &&
    matrix.head_rows() == group &&
    matrix.fold().geometry() == (16, k, d) &&
    threads == 32 &&
    grid_y == kv_heads &&
    heads % kv_heads == 0 &&
    shared >=
    (16L + 2L * k.to_int64() * stages.to_int64()) * d.to_int64() * 2L + 16L else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let partitions = lowering.key_value_partitions()
  let grain = lowering.key_value_partition_grain()
  guard partitions > 0 && grain > 0 && grain % k == 0 else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let fold = matrix.fold()
  guard fold.fragment_products(
      fragment_rows=16,
      fragment_columns=16,
      reduction_quantum=16,
    )
    is Some(program) &&
    @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding::new(
      program,
      subgroup_lanes=32,
      operand_vector_elements=8,
      retained_query_dimensions=d,
    )
    is Some(fragments) else {
    raise InvalidCudaAttentionTileSource(UnsupportedLowering)
  }
  let symbol = if partitions > 1 {
    requested_symbol
  } else {
    requested_symbol + "_grouped_head_matrix_bf16_v1"
  }
  guard valid_cuda_symbol(symbol) else {
    raise InvalidCudaAttentionTileSource(InvalidSymbol)
  }
  let outputs = if partitions > 1 {
    "const __nv_bfloat16 *rotated_qkv_rows, const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache, float *workspace"
  } else {
    "const __nv_bfloat16 *rotated_qkv_rows, __nv_bfloat16 *output, const __nv_bfloat16 *key_cache, const __nv_bfloat16 *value_cache"
  }
  let states = max_rows.to_int64() * heads.to_int64() * partitions.to_int64()
  let initial_states = if partitions > 1 {
    "  if (lane == 0) for (int h=0;h<LF_HEAD_ROWS;++h) {\n" +
    "    const unsigned long long state=((unsigned long long)row*\{heads}ULL+key_value_head*LF_HEAD_ROWS+h)*\{partitions}ULL+blockIdx.z;\n" +
    "    workspace[state]=CUDART_NAN_F; workspace[\{states}ULL+state]=CUDART_NAN_F;\n  }"
  } else {
    ""
  }
  let range = if partitions > 1 {
    "  const int tiles=(query_position+\{grain})/\{grain};\n" +
    "  const int first=(int)((unsigned long long)tiles*blockIdx.z/\{partitions}ULL);\n" +
    "  const int last=(int)((unsigned long long)tiles*(blockIdx.z+1)/\{partitions}ULL);\n" +
    "  const int key_begin=first*\{grain}, key_limit=min(query_position+1,last*\{grain});"
  } else {
    "  const int key_begin=0, key_limit=query_position+1;"
  }
  let terminal = if partitions > 1 {
    "      const unsigned long long state=((unsigned long long)row*\{heads}ULL+query_head)*\{partitions}ULL+blockIdx.z;\n" +
    "      workspace[\{2L * states}ULL+state*LF_HEAD_DIMENSION+component]=out[column].x[item];\n" +
    "      if (component==0) {workspace[state]=maximum[item%4/2];workspace[\{states}ULL+state]=denominator[item%4/2];}"
  } else {
    "      output[((unsigned long long)token*\{heads}ULL+query_head)*LF_HEAD_DIMENSION+component]=__float2bfloat16_rn(out[column].x[item]/denominator[item%4/2]);"
  }
  let transfer = cuda_grouped_matrix_transfer(abi)
  let acquire = if effects.is_pipelined() {
    emit_decode_pipeline_iteration(effects)
  } else {
    cuda_grouped_effects(effects, AcquireKeyOperands, transfer, "    ")
  }
  let initialize = if effects.is_pipelined() {
    emit_decode_async_stage(
      abi,
      effects,
      lowering.hoists_paged_key_value_page_lookup(),
      lowering.decode_copy_retention().unwrap(),
      vector_elements=8,
      tile_rows=k,
      start_alignment=if partitions > 1 { grain } else { k },
      storage_index="lf_operand_index(key, component, LF_HEAD_DIMENSION)",
    )
  } else {
    ""
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <math_constants.h>
    $|// numeric_law=\{matrix.numeric_law()}
    $|#define LF_BLOCK_THREADS 32
    $|#define LF_HEAD_ROWS \{group}
    $|#define LF_KEY_VALUE_TILE \{k}
    $|#define LF_HEAD_DIMENSION \{d}
    $|\{cuda_attention_operand_layout(16, k, d)}
    $|\{cuda_attention_matrix_fragments(shared_scores=false, forwarded_right=fragments)}
    $|\{cuda_query_owned_score_fold()}
    $|\{cuda_query_owned_numeric_primitives(law=fold.exponential_law())}
    $|extern "C" __global__ void \{symbol}(
    $|    const int *counts, const int *query_positions, const int *query_row_offsets,
    $|    const int *sequence_lengths, const int *page_table_offsets,
    $|    const int *page_table_page_indices, \{outputs}) {
    $|  const int thread=threadIdx.x, lane=thread&31, key_value_head=blockIdx.y;
    $|  const int prefill_count=counts[0], decode_count=counts[1], row_count=counts[2];
    $|  const int token_count=counts[3], page_entry_count=counts[4];
    $|  if (blockDim.x!=32 || blockIdx.z>=\{partitions} || key_value_head>=\{kv_heads} ||
    $|      prefill_count<0 || decode_count<=0 || row_count!=prefill_count+decode_count ||
    $|      row_count>\{max_rows} || token_count<=0 || token_count>\{max_tokens} ||
    $|      page_entry_count<=0 || page_entry_count>\{max_pages} || blockIdx.x>=decode_count) return;
    $|  const int row=prefill_count+blockIdx.x;
    $|\{initial_states}
    $|  const int token=query_row_offsets[row], row_end=query_row_offsets[row+1];
    $|  const int sequence_length=sequence_lengths[row];
    $|  const int query_position=token>=0 && token<token_count?query_positions[token]:-1;
    $|  const int page_start=page_table_offsets[row], page_end=page_table_offsets[row+1];
    $|  if(token<0 || row_end!=token+1 || row_end>token_count || query_position<0 ||
    $|     query_position>=sequence_length || page_start<0 || page_start>=page_end ||
    $|     page_end>page_entry_count || page_end-page_start!=(sequence_length+\{page_rows-1})/\{page_rows}) return;
    $|  extern __shared__ __align__(16) unsigned char arena[];
    $|  auto *qbuf=reinterpret_cast<__nv_bfloat16 *>(arena);
    $|  auto *stage_arena=qbuf+16*LF_HEAD_DIMENSION;
    $|  auto *key_tile=stage_arena;
    $|  auto *value_tile=key_tile+LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION;
    $|  int *tile_valid=reinterpret_cast<int *>(stage_arena+\{2*stages}*LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION);
    $|\{if effects.is_pipelined() {"  int *stage_valid=tile_valid;"} else {""}}
    $|  for(int vector=thread;vector<16*LF_HEAD_DIMENSION/8;vector+=32) {
    $|    const int r=vector*8/LF_HEAD_DIMENSION,c=vector*8%LF_HEAD_DIMENSION;
    $|    uint4 packed=make_uint4(0,0,0,0);
    $|    if(r<LF_HEAD_ROWS) packed=*reinterpret_cast<const uint4 *>(rotated_qkv_rows+(unsigned long long)token*\{input_width}ULL+(key_value_head*LF_HEAD_ROWS+r)*LF_HEAD_DIMENSION+c);
    $|    *reinterpret_cast<uint4 *>(qbuf+lf_operand_index(r,c,LF_HEAD_DIMENSION))=packed;
    $|  }
    $|  __syncthreads();
    $|  const int qr=0, active=LF_HEAD_ROWS, p0=lane/4<LF_HEAD_ROWS?query_position:-1;
    $|  const int p1=lane/4+8<LF_HEAD_ROWS?query_position:-1;
    $|  const float scale=rsqrtf((float)LF_HEAD_DIMENSION);
    $|  float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},denominator[2]={};
    $|  LfMatrixAccumulator out[LF_HEAD_DIMENSION/16]={};
    $|\{range}
    $|  const int key_end=key_limit;
    $|\{cuda_query_operand_bootstrap(fragments)}
    $|\{initialize}
    $|  for(int key_base=key_begin;key_base<key_limit;key_base+=LF_KEY_VALUE_TILE) {
    $|    const int kb=key_base;
    $|\{acquire}
    $|    const auto *current_k=key_tile,*current_v=value_tile;
    $|    LfMatrixAccumulator score[LF_KEY_VALUE_TILE/16]={};
    $|\{cuda_online_fold_phase(fold.score_phase(), fragments)}
    $|\{cuda_grouped_effects(effects, ReleaseKeyReaders, "", "    ")}
    $|\{cuda_grouped_effects(effects, AcquireValueOperands, "", "    ")}
    $|\{cuda_online_fold_phase(fold.value_phase(), fragments)}
    $|\{cuda_grouped_effects(effects, ReleaseTileReaders, "", "    ")}
    $|  }
    $|\{if effects.is_pipelined() { cuda_grouped_effects(effects, CompleteTileReaders, "", "  ") } else { "" }}
    $|  #pragma unroll
    $|  for(int column=0;column<LF_HEAD_DIMENSION/16;++column) {
    $|    #pragma unroll
    $|    for(int item=0;item<8;++item) {
    $|      const int h=lf_matrix_row(item,lane),component=column*16+lf_matrix_column(item,lane);
    $|      if(h<LF_HEAD_ROWS) {
    $|        const int query_head=key_value_head*LF_HEAD_ROWS+h;
    $|\{terminal}
    $|      }
    $|    }
    $|  }
    $|}
    $|#undef LF_BLOCK_THREADS
    $|#undef LF_HEAD_ROWS
    $|#undef LF_KEY_VALUE_TILE
    $|#undef LF_HEAD_DIMENSION
  let bytes = @utf8.encode(source)
  guard bytes.length() <= maximum_source_bytes else {
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

///|
/// Cooperative sync alternative shares exactly the same row-swizzled storage
/// as async, making transport and compute independent measurable choices.
fn cuda_grouped_matrix_transfer(abi : CudaPagedAttentionTileAbi) -> String {
  let (_, stride, cache_pages) = abi.layout()
  let (_, kv_heads, d, page_rows) = abi.shape()
  (
    $|    if(thread==0) *tile_valid=1;
    $|    __syncthreads();
    $|    for(int vector=thread;vector<LF_KEY_VALUE_TILE*LF_HEAD_DIMENSION/8;vector+=32) {
    $|      const int linear=vector*8,key=linear/LF_HEAD_DIMENSION,component=linear%LF_HEAD_DIMENSION;
    $|      const int position=key_base+key;
    $|      uint4 kp=make_uint4(0,0,0,0),vp=make_uint4(0,0,0,0);
    $|      if(position<key_limit) {
    $|        const int entry=page_start+position/\{page_rows};
    $|        const int page=entry<page_end?page_table_page_indices[entry]:-1;
    $|        const unsigned long long offset=(unsigned long long)max(page,0)*\{stride}ULL+((unsigned long long)(position%\{page_rows})*\{kv_heads}ULL+key_value_head)*\{d}ULL+component;
    $|        if(page<0 || (unsigned long long)page>=\{cache_pages}ULL) atomicExch(tile_valid,0);
    $|        else {kp=*reinterpret_cast<const uint4 *>(key_cache+offset);vp=*reinterpret_cast<const uint4 *>(value_cache+offset);}
    $|      }
    $|      const int destination=lf_operand_index(key,component,LF_HEAD_DIMENSION);
    $|      *reinterpret_cast<uint4 *>(key_tile+destination)=kp;
    $|      *reinterpret_cast<uint4 *>(value_tile+destination)=vp;
    $|    }
  )
}
