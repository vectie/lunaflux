///|
/// Terminal lowering of the frame-local causal triangular algorithm. No
/// benchmark/reference data, model-name selection or per-token host dispatch.
fn chunk_delta_source(
  plan : @precision.RecurrentDeltaPrecision,
  symbol~ : String,
) -> RecurrentDeltaCudaSource raise {
  let (heads, keys, values) = plan.geometry()
  let (rows, sequences, slots) = plan.capacity()
  let tile = if values < 32 { values } else { 32 }
  let physical = @physical.ChunkDeltaPlan::new(
    plan,
    value_tile=tile,
    maximum_workspace_bytes=9223372036854775807L,
    maximum_local_storage_bytes=49152,
  )
  let per_head = physical.head_floats()
  let operand = rows * keys
  let matrix = rows * rows
  let u_offset = 6 * operand + 3 * matrix
  let base_offset = u_offset + rows * values
  let local_tiles = physical.local_factors()
  let operand_ki = if local_tiles { "k*\{rows+1}+i" } else { "i*\{keys}+k" }
  let operand_kj = if local_tiles { "k*\{rows+1}+j" } else { "j*\{keys}+k" }
  let matrix_pitch = if local_tiles { rows + 1 } else { rows }
  let tile_declaration = if local_tiles {
    "__shared__ float local_k[\{keys*(rows+1)}],local_g[\{keys*(rows+1)}],local_l[\{rows*(rows+1)}],local_inverse[\{rows*(rows+1)}];"
  } else {
    "float *local_k=w+\{operand},*local_g=w+\{2*operand},*local_l=w+\{6*operand},*local_inverse=w+\{6*operand+2*matrix};"
  }
  let tile_store = if local_tiles {
    "local_k[k*\{rows+1}+r]=kn[p]; local_g[k*\{rows+1}+r]=g;"
  } else {
    ""
  }
  let eps = @literal.canonical_cuda_float_literal(plan.normalization_epsilon())
  fn expression(
    op : @precision.RecurrentChunkExpression,
    args : ArrayView[String],
  ) -> String raise {
    @arithmetic.scalar_expression(op.program(), args)
  }
  fn recurrent(
    op : @precision.RecurrentDeltaExpression,
    args : ArrayView[String],
  ) -> String raise {
    @arithmetic.scalar_expression(plan.expression(op), args)
  }
  let square_q = recurrent(SquareUpdate, ["qs", "x"])
  let square_k = recurrent(SquareUpdate, ["ks", "y"])
  let qinv = recurrent(NormInverse, ["qs", eps])
  let kinv = recurrent(NormInverse, ["ks", eps])
  let qscale = recurrent(QueryScale, ["\{keys}.0f"])
  let normalized_q = recurrent(QueryNormalize, ["x", "qi[r]", qscale])
  let normalized_k = recurrent(KeyNormalize, ["y", "ki[r]"])
  let gate = expression(GatePrefix, ["g", "log_decay[i]"])
  let key_first = expression(DecayedOperand, ["kn[p]", "g"])
  let query_first = expression(DecayedOperand, ["qn[p]", "g"])
  let key_last = expression(DecayedOperand, [
    "kn[p]",
    "__fsub_rn(gates[(n-1)*\{keys}+k],gates[p])",
  ])
  // Column-major, padded tile storage gives neighboring pair columns distinct
  // banks while the query row broadcasts. Arithmetic order is unchanged.
  let pair_key = expression(DecayedOperand, [
    "local_k[\{operand_kj}]",
    "__fsub_rn(local_g[\{operand_ki}],local_g[\{operand_kj}])",
  ])
  let lower_sum = expression(ProductUpdate, [
    "l",
    "local_k[\{operand_ki}]",
    "decayed",
  ])
  let query_sum = expression(ProductUpdate, ["p", "qn[i*\{keys}+k]", "decayed"])
  let lower = expression(Scale, [
    "l",
    "__bfloat162float(beta[(begin+i)*\{heads}+head])",
  ])
  let solve = expression(SubtractProduct, [
    "x",
    "local_l[i*\{matrix_pitch}+j]",
    "local_inverse[j*\{matrix_pitch}+column]",
  ])
  let initial_key = expression(ProductUpdate, [
    "read",
    "first_k[r*\{keys}+k]",
    "s",
  ])
  let initial_query = expression(ProductUpdate, [
    "base",
    "first_q[r*\{keys}+k]",
    "s",
  ])
  let rhs = recurrent(Delta, [
    "value[((unsigned long long)(begin+r)*\{heads}+head)*\{values}+v]",
    "read",
    "b",
  ])
  let transform = expression(ProductUpdate, [
    "x",
    "inverse[r*\{rows}+j]",
    "rhs[j]",
  ])
  let projected = expression(ProductUpdate, [
    "x",
    "pair_q[r*\{rows}+j]",
    "transformed[j*\{values}+v]",
  ])
  let state_decay = expression(DecayedOperand, [
    "state[sbase+index]",
    "gates[(n-1)*\{keys}+k]",
  ])
  let state_update = expression(ProductUpdate, [
    "x",
    "last_k[j*\{keys}+k]",
    "transformed[j*\{values}+v]",
  ])
  let output = recurrent(Publish, ["x"])
  let arguments = "const int *counts,const int *offsets,const int *slots,const int *reset,const __nv_bfloat16 *query,const __nv_bfloat16 *key,const __nv_bfloat16 *value,const float *log_decay,const __nv_bfloat16 *beta,float *state,__nv_bfloat16 *output,float *scratch"
  let validation =
    $|  const int sequence=blockIdx.x, head=blockIdx.y, lane=threadIdx.x;
    $|  if(counts[2]<=0||counts[2]>\{sequences}||counts[3]<=0||counts[3]>\{rows}||sequence>=counts[2]||head>=\{heads}) return;
    $|  const int begin=offsets[sequence],end=offsets[sequence+1],slot=slots[sequence],n=end-begin;
    $|  if(begin<0||end>counts[3]||n<=0||slot<0||slot>=\{slots}) return;
    $|  float *w=scratch+((unsigned long long)sequence*\{heads}+head)*\{per_head}ULL;
  let text =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|extern "C" __global__ void \{symbol}_prepare(\{arguments}) {
    $|\{validation}
    $|  __shared__ float qi[\{rows}],ki[\{rows}];
    $|  \{tile_declaration}
    $|  float *qn=w,*kn=w+\{operand},*gates=w+\{2*operand},*first_k=w+\{3*operand},*first_q=w+\{4*operand},*last_k=w+\{5*operand};
    $|  for(int r=lane;r<n;r+=blockDim.x) {
    $|    float qs=0.0f,ks=0.0f;
    $|    for(int k=0;k<\{keys};++k) {
    $|      const unsigned long long i=((unsigned long long)(begin+r)*\{heads}+head)*\{keys}+k;
    $|      const float x=__bfloat162float(query[i]),y=__bfloat162float(key[i]);
    $|      qs=\{square_q}; ks=\{square_k};
    $|    }
    $|    qi[r]=\{qinv}; ki[r]=\{kinv};
    $|  }
    $|  __syncthreads();
    $|  for(int k=lane;k<\{keys};k+=blockDim.x) {
    $|    float g=0.0f;
    $|    for(int r=0;r<n;++r) {
    $|      const unsigned long long i=((unsigned long long)(begin+r)*\{heads}+head)*\{keys}+k;
    $|      const int p=r*\{keys}+k;
    $|      const float x=__bfloat162float(query[i]),y=__bfloat162float(key[i]);
    $|      qn[p]=\{normalized_q}; kn[p]=\{normalized_k};
    $|      g=\{gate}; gates[p]=g;
    $|      \{tile_store}
    $|      first_k[p]=\{key_first}; first_q[p]=\{query_first};
    $|    }
    $|    for(int r=0;r<n;++r) { const int p=r*\{keys}+k; last_k[p]=\{key_last}; }
    $|  }
    $|  __syncthreads();
    $|  float *lower=w+\{6*operand},*pair_q=lower+\{matrix},*inverse=pair_q+\{matrix};
    $|  for(int cell=lane;cell<n*n;cell+=blockDim.x) {
    $|    const int i=cell/n,j=cell%n; float l=0.0f,p=0.0f;
    $|    if(j<=i) for(int k=0;k<\{keys};++k) {
    $|      const float decayed=\{pair_key};
    $|      l=\{lower_sum}; p=\{query_sum};
    $|    }
    $|    lower[i*\{rows}+j]=j<i ? \{lower} : 0.0f;
    $|    local_l[i*\{matrix_pitch}+j]=lower[i*\{rows}+j];
    $|    pair_q[i*\{rows}+j]=p;
    $|  }
    $|  __syncthreads();
    $|  // Independent columns of the unit-lower triangular inverse.
    $|  for(int column=lane;column<n;column+=blockDim.x) {
    $|    for(int i=0;i<n;++i) {
    $|      float x=i==column ? 1.0f : 0.0f;
    $|      for(int j=column;j<i;++j) x=\{solve};
    $|      local_inverse[i*\{matrix_pitch}+column]=x;
    $|    }
    $|  }
    $|  __syncthreads();
    $|  for(int cell=lane;cell<n*n;cell+=blockDim.x) inverse[(cell/n)*\{rows}+cell%n]=local_inverse[(cell/n)*\{matrix_pitch}+cell%n];
    $|}
    $|extern "C" __global__ void \{symbol}_solve(\{arguments}) {
    $|\{validation}
    $|  const int v=blockIdx.z*\{tile}+lane; if(lane>=\{tile}||v>=\{values}) return;
    $|  const unsigned long long sbase=((unsigned long long)slot*\{heads}+head)*\{keys}*\{values};
    $|  const float *first_k=w+\{3*operand},*first_q=w+\{4*operand},*inverse=w+\{6*operand+2*matrix};
    $|  float *transformed=w+\{u_offset},*base_output=w+\{base_offset};
    $|  float rhs[\{rows}];
    $|  for(int r=0;r<n;++r) {
    $|    float read=0.0f,base=0.0f;
    $|    if(!reset[sequence]) for(int k=0;k<\{keys};++k) {
    $|      const float s=state[sbase+(unsigned long long)k*\{values}+v];
    $|      read=\{initial_key}; base=\{initial_query};
    $|    }
    $|    const float b=__bfloat162float(beta[(begin+r)*\{heads}+head]);
    $|    rhs[r]=\{rhs}; base_output[r*\{values}+v]=base;
    $|  }
    $|  for(int r=0;r<n;++r) {
    $|    float x=0.0f; for(int j=0;j<=r;++j) x=\{transform};
    $|    transformed[r*\{values}+v]=x;
    $|  }
    $|}
    $|extern "C" __global__ void \{symbol}_publish(\{arguments}) {
    $|\{validation}
    $|  const int index=blockIdx.z*128+lane;
    $|  const float *gates=w+\{2*operand},*last_k=w+\{5*operand},*pair_q=w+\{6*operand+matrix},*transformed=w+\{u_offset},*base_output=w+\{base_offset};
    $|  if(index<n*\{values}) {
    $|    const int r=index/\{values},v=index%\{values}; float x=base_output[index];
    $|    for(int j=0;j<=r;++j) x=\{projected};
    $|    output[((unsigned long long)(begin+r)*\{heads}+head)*\{values}+v]=\{output};
    $|  }
    $|  if(index<\{keys*values}) {
    $|    const int k=index/\{values},v=index%\{values};
    $|    const unsigned long long sbase=((unsigned long long)slot*\{heads}+head)*\{keys}*\{values};
    $|    float x=reset[sequence] ? 0.0f : \{state_decay};
    $|    for(int j=0;j<n;++j) x=\{state_update};
    $|    state[sbase+index]=x;
    $|  }
    $|}
  {
    plan,
    symbol,
    source: @utf8.encode(text),
    block: 128,
    chunk: Some(physical),
  }
}
