///|
/// Slot identity is not an address. The staged sequence's scheduler page list
/// selects every append address; page generations distinguish reused storage.
/// Copy and valid publication are separate ordered effects, as for contiguous
/// history. No grid synchronization or per-request full-pool clearing occurs.
fn paged_history_source(
  plan : @precision.IndexedSparseHistoryPrecision,
  symbols : ArrayView[String],
) -> Bytes {
  let attention = plan.attention()
  let domain = attention.paged_history().unwrap()
  let (page_rows, pages, context, sequences) = domain.geometry()
  let (rows, _) = attention.capacity()
  let (_, dimension, _, _, _) = attention.index_geometry()
  let (_, heads, keys, values) = attention.attention_geometry()
  let kw = heads.to_int64() * keys.to_int64()
  let vw = if attention.shares_key_value() {
    0L
  } else {
    heads.to_int64() * values.to_int64()
  }
  let width = dimension.to_int64() * 2L + kw + vw
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbols[0]}(const int32_t *counts,const int32_t *reset,int32_t *history,int32_t *valid,int32_t *append,int32_t *positions,const int32_t *input_positions,const int32_t *row_sequences,const int32_t *page_offsets,const int32_t *page_ids,const int32_t *page_generations) {
    $| if(blockIdx.x)return;
    $| const int live=counts[3],n=counts[2];
    $| if(threadIdx.x==0){append[0]=0;append[1]=live;append[2]=(live<0||live>\{rows}||n<0||n>\{sequences}||(live>0&&n==0));history[0]=counts[0];history[1]=counts[1];history[2]=n;history[3]=\{domain.physical_rows()};history[4]=0;}
    $| __syncthreads();
    $| for(int row=threadIdx.x;row<\{rows};row+=blockDim.x){
    $|  positions[row]=-1;append[3+row*2]=-1;append[4+row*2]=0;
    $|  if(row>=live||live>\{rows}||n<=0||n>\{sequences})continue;
    $|  const int sequence=row_sequences[row],position=input_positions[row];
    $|  if(sequence<0||sequence>=n||position<0||position>=\{context}){atomicOr(append+2,1);continue;}
    $|  const int first=page_offsets[sequence],end=page_offsets[sequence+1];
    $|  if(first<0||end<first||end>\{domain.table_capacity()}||position/\{page_rows}>=end-first){atomicOr(append+2,1);continue;}
    $|  const int at=first+position/\{page_rows},page=page_ids[at],generation=page_generations[at];
    $|  if(page<0||page>=\{pages}||generation<=0){atomicOr(append+2,1);continue;}
    $|  positions[row]=position;append[3+row*2]=page*\{page_rows}+position%\{page_rows};append[4+row*2]=generation;
    $| }
    $|}
    $|extern "C" __global__ void \{symbols[1]}(const int32_t *append,const __nv_bfloat16 *index_keys,const __nv_bfloat16 *gates,const __nv_bfloat16 *keys,const __nv_bfloat16 *values,__nv_bfloat16 *history_index,__nv_bfloat16 *history_gates,__nv_bfloat16 *history_keys,__nv_bfloat16 *history_values) {
    $| if(append[2])return;
    $| const uint64_t cell=(uint64_t)blockIdx.x*256ULL+threadIdx.x,row=cell/\{width}ULL,component=cell%\{width}ULL;
    $| if(row>=(uint64_t)append[1])return;
    $| const uint64_t destination=append[3+row*2];
    $| if(component<\{dimension}ULL)history_index[destination*\{dimension}ULL+component]=index_keys[row*\{dimension}ULL+component];
    $| else if(component<\{dimension*2}ULL){uint64_t i=component-\{dimension}ULL;history_gates[destination*\{dimension}ULL+i]=gates[row*\{dimension}ULL+i];}
    $| else if(component<\{dimension.to_int64()*2L+kw}ULL){uint64_t i=component-\{dimension*2}ULL;history_keys[destination*\{kw}ULL+i]=keys[row*\{kw}ULL+i];}
    $| else {uint64_t i=component-\{dimension.to_int64()*2L+kw}ULL;history_values[destination*\{vw}ULL+i]=values[row*\{vw}ULL+i];}
    $|}
    $|extern "C" __global__ void \{symbols[2]}(const int32_t *append,int32_t *history,int32_t *valid) {
    $| const int row=blockIdx.x*blockDim.x+threadIdx.x;
    $| if(!append[2]&&row<append[1]&&row<\{rows})valid[append[3+row*2]]=append[4+row*2];
    $|}
  @utf8.encode(source)
}
