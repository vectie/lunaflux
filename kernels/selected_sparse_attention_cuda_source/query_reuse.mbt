///|
/// Device lowering for shared history reads across a query cohort. The dense
/// physical traversal is masked by each row's original sparse selection.
/// BF16 probability publication remains between the two matrix products.
fn query_reuse_source(
  qh : Int,
  kvh : Int,
  qkd : Int,
  vd : Int,
  width : Int,
  extent : Int,
  rows : Int,
  history : Int,
  symbol : String,
  paging : @physical.PagedHistoryDomain?,
  score_dimension : Int,
) -> Bytes {
  let words = (history + 31) / 32
  let rounds = (vd + 63) / 64
  let storage = 16 * Int::max(qkd, vd)
  let ports = if paging is Some(_) {
    ",const int32_t *row_sequences,const int32_t *page_offsets,const int32_t *page_ids,const int32_t *page_generations"
  } else {
    ""
  }
  let (live, sequence, count_guard, address) = match paging {
    None =>
      (
        "*key_count", "0", "counts[2]!=1", "return key>=0&&key<live&&key_valid[key]!=0?key:-1;",
      )
    Some(domain) => {
      let (page_rows, pages, _, sequences) = domain.geometry()
      (
        "\{history}",
        "row_sequences[query]",
        "counts[2]<=0||counts[2]>\{sequences}",
        (
          $|if(key<0||key>=live||sequence<0||sequence>=counts[2])return -1;
          $|int first=page_offsets[sequence],end=page_offsets[sequence+1],at=first+key/\{page_rows};
          $|if(first<0||end<first||end>\{domain.table_capacity()}||at>=end)return -1;
          $|int page=page_ids[at];if(page<0||page>=\{pages}||page_generations[at]<=0)return -1;
          $|int physical=page*\{page_rows}+key%\{page_rows};
          $|return key_valid[physical]==page_generations[at]?physical:-1;
        ),
      )
    }
  }
  let source =
    $|#include <cuda_bf16.h>
    $|#include <stdint.h>
    $|#include <math.h>
    $|__device__ __forceinline__ unsigned \{symbol}_pack(__nv_bfloat16 a,__nv_bfloat16 b){return (unsigned)__bfloat16_as_ushort(a)|((unsigned)__bfloat16_as_ushort(b)<<16);}
    $|__device__ __forceinline__ void \{symbol}_matrix(unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,float *d){
    $| asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    $|}
    $|__device__ __forceinline__ void \{symbol}(const int32_t *counts,const int32_t *query_positions,const int32_t *key_count,const int32_t *key_valid,const __nv_bfloat16 *queries,const __nv_bfloat16 *keys,const __nv_bfloat16 *values,const int32_t *selected_indices,__nv_bfloat16 *output,void *scratch\{ports},unsigned char *branch_storage){
    $| const int worker=threadIdx.x,lane=worker&31,warp=worker/32,g=lane/4,t=lane%4;
    $| const int head=blockIdx.x%\{qh},first_query=(blockIdx.x/\{qh})*16,kv_head=head/\{qh/kvh};
    $| if(first_query>=\{rows})return;
    $| const int live=\{live};const __nv_bfloat16 zero=__float2bfloat16_rn(0.0f);
    $| unsigned (*mask)[\{words}]=(unsigned (*)[\{words}])branch_storage;
    $| __shared__ int valid[16],sequence_id[16],position[16],physical[64],limit,any;
    $| __shared__ __nv_bfloat16 operands[\{storage}],probabilities[16][16];
    $| for(int i=worker;i<16*\{vd};i+=256){int r=i/\{vd},c=i%\{vd},query=first_query+r;if(query<\{rows})output[((uint64_t)query*\{qh}+head)*\{vd}+c]=zero;}
    $| for(int i=worker;i<16*\{words};i+=256)mask[i/\{words}][i%\{words}]=0U;
    $| if(worker<16){
    $|  int query=first_query+worker;valid[worker]=0;sequence_id[worker]=-1;position[worker]=-1;
    $|  if(!(\{count_guard})&&counts[3]>0&&counts[3]<=\{rows}&&query<counts[3]&&live>0&&live<=\{history}){
    $|   int seq=\{sequence},p=query_positions[query];if(seq>=0&&seq<counts[2]&&p>=0&&p<live){valid[worker]=1;sequence_id[worker]=seq;position[worker]=p;}
    $|  }
    $| }
    $| __syncthreads();
    $| auto physical_key=[&](int sequence,int key)->int{\{address}};
    $| for(int i=worker;i<16*\{extent};i+=256){
    $|  int r=i/\{extent},rank=i%\{extent},query=first_query+r;
    $|  if(sequence_id[r]<0)continue;
    $|  int key=selected_indices[(uint64_t)query*\{width}+rank];if(key==-1)continue;
    $|  if(key<0||key>position[r]||physical_key(sequence_id[r],key)<0){atomicExch(&valid[r],0);continue;}
    $|  unsigned bit=1U<<(key&31);if(atomicOr(&mask[r][key/32],bit)&bit)atomicExch(&valid[r],0);
    $| }
    $| __syncthreads();
    $| for(int leader=0;leader<16;++leader){
    $|  int seq=sequence_id[leader];bool owns=valid[leader]!=0;
    $|  for(int r=0;r<leader;++r)if(valid[r]&&sequence_id[r]==seq)owns=false;
    $|  if(!owns)continue;
    $|  if(worker==0){limit=0;for(int r=0;r<16;++r)if(valid[r]&&sequence_id[r]==seq)limit=max(limit,position[r]+1);}
    $|  for(int i=worker;i<16*\{qkd};i+=256){int r=i/\{qkd},c=i%\{qkd},query=first_query+r;operands[i]=(valid[r]&&sequence_id[r]==seq)?queries[((uint64_t)query*\{qh}+head)*\{qkd}+c]:zero;}
    $|  __syncthreads();
    $|  for(int base=0;base<limit;base+=64){
    $|   if(worker<64)physical[worker]=base+worker<limit?physical_key(seq,base+worker):-1;
    $|   __syncthreads();
    $|   int key=base+warp*8+g,raw=physical[warp*8+g];float d[4]={0,0,0,0};
    $|   for(int c=0;c<\{qkd};c+=16){
    $|    int a=c+t*2;
    $|    unsigned a0=\{symbol}_pack(operands[g*\{qkd}+a],operands[g*\{qkd}+a+1]),a1=\{symbol}_pack(operands[(g+8)*\{qkd}+a],operands[(g+8)*\{qkd}+a+1]);
    $|    unsigned a2=\{symbol}_pack(operands[g*\{qkd}+a+8],operands[g*\{qkd}+a+9]),a3=\{symbol}_pack(operands[(g+8)*\{qkd}+a+8],operands[(g+8)*\{qkd}+a+9]);
    $|    uint64_t kb=((uint64_t)(raw<0?0:raw)*\{kvh}+kv_head)*\{qkd}+a;
    $|    unsigned b0=raw>=0?\{symbol}_pack(keys[kb],keys[kb+1]):0U,b1=raw>=0?\{symbol}_pack(keys[kb+8],keys[kb+9]):0U;
    $|    \{symbol}_matrix(a0,a1,a2,a3,b0,b1,d);
    $|   }
    $|   for(int rr=0;rr<2;++rr)for(int cc=0;cc<2;++cc){
    $|    int r=g+rr*8,k=base+warp*8+t*2+cc,query=first_query+r;
    $|    if(valid[r]&&sequence_id[r]==seq&&k<limit)((float*)scratch)[((uint64_t)query*\{qh}+head)*\{history}+k]=((mask[r][k/32]>>(k&31))&1U)?__fmul_rn(d[rr*2+cc],rsqrtf(\{score_dimension}.0f)):0.0f;
    $|   }
    $|   __syncthreads();
    $|  }
    $|  // Explicit tiled-F32 law permits parallel sums; P remains BF16.
    $|  for(int r=warp;r<16;r+=8)if(valid[r]&&sequence_id[r]==seq){
    $|   int query=first_query+r;float *s=((float*)scratch)+((uint64_t)query*\{qh}+head)*\{history};
    $|   float maximum=-INFINITY,denominator=0.0f;
    $|   for(int rank=lane;rank<\{extent};rank+=32){int k=selected_indices[(uint64_t)query*\{width}+rank];if(k>=0)maximum=fmaxf(maximum,s[k]);}
    $|   for(int offset=16;offset;offset/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffU,maximum,offset));
    $|   for(int rank=lane;rank<\{extent};rank+=32){int k=selected_indices[(uint64_t)query*\{width}+rank];if(k>=0)denominator=__fadd_rn(denominator,expf(__fsub_rn(s[k],maximum)));}
    $|   for(int offset=16;offset;offset/=2)denominator=__fadd_rn(denominator,__shfl_xor_sync(0xffffffffU,denominator,offset));
    $|   for(int rank=lane;rank<\{extent};rank+=32){int k=selected_indices[(uint64_t)query*\{width}+rank];if(k>=0)s[k]=__bfloat162float(__float2bfloat16_rn(__fdiv_rn(expf(__fsub_rn(s[k],maximum)),denominator)));}
    $|  }
    $|  __syncthreads(); // query operands retire before value staging reuses them
    $|  float accum[\{rounds}][4]={};
    $|  for(int base=0;base<limit;base+=16){
    $|   if(worker==0){any=0;for(int r=0;r<16;++r)if(valid[r]&&sequence_id[r]==seq)any|=(mask[r][base/32]>>(base&31))&65535U;}
    $|   if(worker<16)physical[worker]=base+worker<limit?physical_key(seq,base+worker):-1;
    $|   __syncthreads();
    $|   const int active_tile=any;
    $|   __syncthreads(); // retire every flag reader, including an empty tile
    $|   if(!active_tile)continue;
    $|   for(int i=worker;i<16*\{vd};i+=256){int r=i/\{vd},c=i%\{vd},raw=physical[r];operands[i]=raw>=0?values[((uint64_t)raw*\{kvh}+kv_head)*\{vd}+c]:zero;}
    $|   if(worker<256){int r=worker/16,k=base+worker%16,query=first_query+r;probabilities[r][worker%16]=(valid[r]&&sequence_id[r]==seq&&k<limit)?__float2bfloat16_rn(((float*)scratch)[((uint64_t)query*\{qh}+head)*\{history}+k]):zero;}
    $|   __syncthreads();
    $|   unsigned a0=\{symbol}_pack(probabilities[g][t*2],probabilities[g][t*2+1]),a1=\{symbol}_pack(probabilities[g+8][t*2],probabilities[g+8][t*2+1]);
    $|   unsigned a2=\{symbol}_pack(probabilities[g][t*2+8],probabilities[g][t*2+9]),a3=\{symbol}_pack(probabilities[g+8][t*2+8],probabilities[g+8][t*2+9]);
    $|   for(int round=0;round<\{rounds};++round){int column=round*64+warp*8+g;
    $|    unsigned b0=column<\{vd}?\{symbol}_pack(operands[t*2*\{vd}+column],operands[(t*2+1)*\{vd}+column]):0U;
    $|    unsigned b1=column<\{vd}?\{symbol}_pack(operands[(t*2+8)*\{vd}+column],operands[(t*2+9)*\{vd}+column]):0U;
    $|    \{symbol}_matrix(a0,a1,a2,a3,b0,b1,accum[round]);
    $|   }
    $|   __syncthreads(); // every matrix consumer retires before the next value tile
    $|  }
    $|  for(int round=0;round<\{rounds};++round)for(int rr=0;rr<2;++rr)for(int cc=0;cc<2;++cc){
    $|   int r=g+rr*8,c=round*64+warp*8+t*2+cc,query=first_query+r;
    $|   if(valid[r]&&sequence_id[r]==seq&&c<\{vd})output[((uint64_t)query*\{qh}+head)*\{vd}+c]=__float2bfloat16_rn(accum[round][rr*2+cc]);
    $|  }
    $|  __syncthreads();
    $| }
    $|}
  @utf8.encode(source)
}
