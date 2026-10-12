///|
// CUDA-only physical ownership: sixteen query heads share one KV tile. The
// semantic plan explicitly permits F32 reassociation, without naming an ISA.
fn matrix_window_attention_source(
  plan : @precision.WindowSharedKvPrecision,
  symbol_prefix~ : String,
  maximum_slots~ : Int,
  logical_slots~ : String,
  setup~ : String,
  lookup~ : String,
) -> Bytes raise {
  guard plan.accumulation_law() ==
    @precision.AttentionAccumulationLaw::TiledFloat32 &&
    plan.probability_law() is OnlineBf16(block_tokens~) else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (heads, width, window, rows, _) = plan.geometry()
  guard width >= 16 &&
    width <= 512 &&
    width % 16 == 0 &&
    block_tokens >= 32 &&
    block_tokens <= 128 &&
    maximum_slots > 0 &&
    maximum_slots <= 2147483519 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let output_windows = (width + 63) / 64
  let scale = (1.0 / width.to_double().sqrt()).to_string() + "f"
  @utf8.encode(
    (
      $|#include <mma.h>
      $|extern "C" __global__ void \{symbol_prefix}_attention(\{setup}) {
      $|  namespace wm = nvcuda::wmma;
      $|  const int row=blockIdx.x, head_first=blockIdx.y*16, tid=threadIdx.x, warp=tid/32, lane=tid%32;
      $|  if(row>=\{rows} || head_first>=\{heads}) return;
      $|  if(append[2] || row>=append[1]) {
      $|    for(int i=tid;i<16*\{width};i+=256) { const int h=head_first+i/\{width}; if(h<\{heads}) output[((unsigned long long)row*\{heads}+h)*\{width}+i%\{width}]=__float2bfloat16_rn(0.0f); }
      $|    return;
      $|  }
      $|  const int base=append[0], position=base+row;
      $|  const int first=position>=\{window-1}?position-\{window-1}:0, length=position-first+1;
      $|  const int window_span=base==0?min(\{window},append[1]):\{window};
      $|  const int slots=\{logical_slots};
      $|  __shared__ __align__(32) __nv_bfloat16 qa[16*16], kv[\{block_tokens}*16], probabilities[16*\{block_tokens}];
      $|  __shared__ __align__(32) __nv_bfloat16 vb[16*64];
      $|  __shared__ __align__(32) float scores[16*\{block_tokens}], fragments[4*16*16];
      $|  __shared__ float partial[512], maxima[16], totals[16], alpha[16];
      $|  if(tid<16) { maxima[tid]=-INFINITY; totals[tid]=0.0f; }
      $|  wm::fragment<wm::accumulator,16,16,16,float> accumulated[\{output_windows}];
      $|  if(warp<4) for(int w=0;w<\{output_windows};++w) wm::fill_fragment(accumulated[w],0.0f);
      $|  __syncthreads();
      $|  for(int block=0;block<(slots+\{block_tokens-1})/\{block_tokens};++block) {
      $|    wm::fragment<wm::accumulator,16,16,16,float> score_fragment;
      $|    if(warp<\{block_tokens/16}) wm::fill_fragment(score_fragment,0.0f);
      $|    for(int chunk=0;chunk<\{width};chunk+=16) {
      $|      for(int i=tid;i<256;i+=256) { const int h=head_first+i/16; qa[i]=h<\{heads}?queries[((unsigned long long)row*\{heads}+h)*\{width}+chunk+i%16]:__float2bfloat16_rn(0.0f); }
      $|      for(int i=tid;i<\{block_tokens}*16;i+=256) {
      $|        const int ordinal=block*\{block_tokens}+i/16; const __nv_bfloat16 *key=nullptr;
      $|        if(ordinal<slots) {
      $|\{lookup}
      $|        }
      $|        kv[i]=key?key[chunk+i%16]:__float2bfloat16_rn(0.0f);
      $|      }
      $|      __syncthreads();
      $|      if(warp<\{block_tokens/16}) {
      $|        wm::fragment<wm::matrix_a,16,16,16,__nv_bfloat16,wm::row_major> a;
      $|        wm::fragment<wm::matrix_b,16,16,16,__nv_bfloat16,wm::col_major> b;
      $|        wm::load_matrix_sync(a,qa,16); wm::load_matrix_sync(b,kv+warp*256,16);
      $|        wm::mma_sync(score_fragment,a,b,score_fragment);
      $|      }
      $|      __syncthreads();
      $|    }
      $|    if(warp<\{block_tokens/16}) wm::store_matrix_sync(scores+warp*16,score_fragment,\{block_tokens},wm::mem_row_major);
      $|    __syncthreads();
      $|    // Two head rows per thread, four lanes per head, eight warps. This
      $|    // publishes max/sum reductions independently of matrix fragments.
      $|    for(int r=0;r<2;++r) {
      $|      const int h=lane/4+r*8;
      $|      float value=-INFINITY;
      $|      for(int j=0;j<\{block_tokens/32};++j) {
      $|        const int col=warp*\{block_tokens/8}+(lane%4)*\{block_tokens/32}+j, ordinal=block*\{block_tokens}+col;
      $|        const __nv_bfloat16 *key=nullptr;
      $|        if(ordinal<slots) {
      $|\{lookup}
      $|        }
      $|        const float score=key?__fmul_rn(scores[h*\{block_tokens}+col],\{scale}):-INFINITY;
      $|        scores[h*\{block_tokens}+col]=score; value=fmaxf(value,score);
      $|      }
      $|      partial[r*256+tid]=value;
      $|    }
      $|    __syncthreads();
      $|    for(int stride=128;stride>=32;stride/=2) { if(tid<stride) for(int r=0;r<2;++r) partial[r*256+tid]=fmaxf(partial[r*256+tid],partial[r*256+tid+stride]); __syncthreads(); }
      $|    if(tid<32) for(int r=0;r<2;++r) {
      $|      float value=partial[r*256+tid]; value=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,2)); value=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
      $|      if(lane%4==0) { const int h=lane/4+r*8; const float next=fmaxf(maxima[h],value); alpha[h]=next==-INFINITY?1.0f:(maxima[h]==-INFINITY?0.0f:expf(__fsub_rn(maxima[h],next))); maxima[h]=next; }
      $|    }
      $|    __syncthreads();
      $|    for(int r=0;r<2;++r) {
      $|      const int h=lane/4+r*8; float sum=0.0f;
      $|      for(int j=0;j<\{block_tokens/32};++j) {
      $|        const int col=warp*\{block_tokens/8}+(lane%4)*\{block_tokens/32}+j;
      $|        const float p=maxima[h]==-INFINITY?0.0f:expf(__fsub_rn(scores[h*\{block_tokens}+col],maxima[h]));
      $|        probabilities[h*\{block_tokens}+col]=__float2bfloat16_rn(p); sum=__fadd_rn(sum,p);
      $|      }
      $|      partial[r*256+tid]=sum;
      $|    }
      $|    __syncthreads();
      $|    for(int stride=128;stride>=32;stride/=2) { if(tid<stride) for(int r=0;r<2;++r) partial[r*256+tid]=__fadd_rn(partial[r*256+tid],partial[r*256+tid+stride]); __syncthreads(); }
      $|    if(tid<32) for(int r=0;r<2;++r) { float sum=partial[r*256+tid]; sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,2)); sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,1)); if(lane%4==0) { const int h=lane/4+r*8; totals[h]=__fmaf_rn(totals[h],alpha[h],sum); } }
      $|    __syncthreads();
      $|    // Rescale in an explicit row-owned F32 domain; never guess the
      $|    // vendor fragment register-to-row mapping or round the accumulator.
      $|    for(int w=0;w<\{output_windows};++w) {
      $|      if(warp<4) wm::store_matrix_sync(fragments+warp*256,accumulated[w],16,wm::mem_row_major);
      $|      __syncthreads();
      $|      if(warp<4) for(int i=lane;i<256;i+=32) fragments[warp*256+i]=__fmul_rn(fragments[warp*256+i],alpha[i/16]);
      $|      __syncthreads();
      $|      if(warp<4) wm::load_matrix_sync(accumulated[w],fragments+warp*256,16,wm::mem_row_major);
      $|      __syncthreads();
      $|    }
      $|    for(int chunk=0;chunk<\{block_tokens};chunk+=16) {
      $|      for(int w=0;w<\{output_windows};++w) {
      $|        for(int i=tid;i<16*64;i+=256) {
      $|          const int ordinal=block*\{block_tokens}+chunk+i/64, col=w*64+i%64; const __nv_bfloat16 *key=nullptr;
      $|          if(ordinal<slots) {
      $|\{lookup}
      $|          }
      $|          vb[i]=key&&col<\{width}?key[col]:__float2bfloat16_rn(0.0f);
      $|        }
      $|        __syncthreads();
      $|        if(warp<4) {
      $|          wm::fragment<wm::matrix_a,16,16,16,__nv_bfloat16,wm::row_major> a;
      $|          wm::fragment<wm::matrix_b,16,16,16,__nv_bfloat16,wm::row_major> b;
      $|          wm::load_matrix_sync(a,probabilities+chunk,\{block_tokens}); wm::load_matrix_sync(b,vb+warp*16,64);
      $|          wm::mma_sync(accumulated[w],a,b,accumulated[w]);
      $|        }
      $|        __syncthreads();
      $|      }
      $|    }
      $|  }
      $|  if(tid<16) totals[tid]=__fadd_rn(totals[tid],expf(__fsub_rn(head_first+tid<\{heads}?sink[head_first+tid]:0.0f,maxima[tid])));
      $|  __syncthreads();
      $|  for(int w=0;w<\{output_windows};++w) {
      $|    if(warp<4) wm::store_matrix_sync(fragments+warp*256,accumulated[w],16,wm::mem_row_major);
      $|    __syncthreads();
      $|    if(warp<4) for(int i=lane;i<256;i+=32) { const int h=head_first+i/16, col=w*64+warp*16+i%16; if(h<\{heads} && col<\{width}) output[((unsigned long long)row*\{heads}+h)*\{width}+col]=__float2bfloat16_rn(__fdiv_rn(fragments[warp*256+i],totals[i/16])); }
      $|    __syncthreads();
      $|  }
      $|}
    ),
  )
}
