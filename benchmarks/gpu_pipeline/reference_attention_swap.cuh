// Offline, fixed-workload causal adapter. Never a production dispatch policy.
// Uses the pinned vLLM FlashAttention-2 inference implementation. Adapter work,
// dynamic descriptor reads, and extra no-op launches are INCLUDED in timings.
#include <optional>
#include "flash.h"
#include "flash_fwd_kernel.h"
using RefTraits=Flash_fwd_kernel_traits<128,64,128,4,false,false,cutlass::bfloat16_t>;
static_assert(RefTraits::kSmemSize==81920);
struct AttentionArgs {
  const int *counts=nullptr,*positions=nullptr,*offsets=nullptr,*lengths=nullptr;
  const int *page_offsets=nullptr,*pages=nullptr;
  const __nv_bfloat16 *q=nullptr,*k=nullptr,*v=nullptr;
  const int *padded_pages=nullptr;
};
// The pinned reference resolver can read a page-table slot past the logical
// final page for predicated-off vector lanes. Its host ABI provides padding;
// LunaFlux's CSR table does not. Materialize a bounded rectangular adapter.
static constexpr int reference_page_slots=1040; // ceil((8192+64)/128)*16 pages
__global__ void reference_pad_page_table(AttentionArgs a,int* table) {
  int row=blockIdx.x;
  if(row>=a.counts[2]) return;
  int begin=a.page_offsets[row],n=a.page_offsets[row+1]-begin;
  for(int i=threadIdx.x;i<reference_page_slots;i+=blockDim.x)
    table[row*reference_page_slots+i]=i<n?a.pages[begin+i]:0;
}
__device__ flash::Flash_fwd_params ref_parameters(AttentionArgs a,__nv_bfloat16* out) {
  flash::Flash_fwd_params p{};
  p.q_ptr=const_cast<__nv_bfloat16*>(a.q);
  p.k_ptr=const_cast<__nv_bfloat16*>(a.k);p.v_ptr=const_cast<__nv_bfloat16*>(a.v);
  p.o_ptr=out;p.d=128;p.d_rounded=128;p.is_bf16=true;
  p.k_batch_stride=8192;p.v_batch_stride=8192;
  p.k_row_stride=1024;p.v_row_stride=1024;
  p.k_head_stride=128;p.v_head_stride=128;
  p.page_block_size=8;p.scale_softmax=0.08838834764831845f;
  p.scale_softmax_log2=p.scale_softmax*1.4426950408889634f;
  p.p_dropout=1.f;p.rp_dropout=1.f;p.scale_softmax_rp_dropout=p.scale_softmax;
  p.window_size_left=-1;p.window_size_right=0;
  return p;
}
__global__ void reference_mixed_attention(AttentionArgs a,__nv_bfloat16* out) {
  if(a.counts[0]==0 || a.counts[2]<1 || a.counts[2]>8 || a.counts[3]>2048) return;
  const int row=blockIdx.y;
  if(row>=a.counts[2]) return;
  const int begin=a.offsets[row],n=a.offsets[row+1]-begin;
  if(n<=0 || int(blockIdx.x)*64>=n) return;
  if(a.positions[begin]!=a.lengths[row]-n || a.positions[begin+n-1]!=a.lengths[row]-1) return;
  auto p=ref_parameters(a,out+size_t(begin)*2048);
  p.q_ptr=const_cast<__nv_bfloat16*>(a.q+size_t(begin)*4096);
  p.q_row_stride=4096;p.q_head_stride=128;p.o_row_stride=2048;p.o_head_stride=128;
  p.b=1;p.h=16;p.h_k=8;p.h_h_k_ratio=2;
  p.seqlen_q=n;p.seqlen_q_rounded=(n+63)/64*64;p.total_q=n;
  p.seqlen_k=a.lengths[row];p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
  p.block_table=const_cast<int*>(a.padded_pages+row*reference_page_slots);
  p.is_causal=true;p.num_splits=1;
  flash::compute_attn_1rowblock_splitkv<RefTraits,true,false,false,false,true,false,false,false>(p,0,blockIdx.z,blockIdx.x,0,1);
}
// Measured C8 baseline: GQA heads reinterpreted as two query rows, 3 KV
// partitions, then the upstream 4-row / log2-max-splits=2 merge kernel.
__global__ void reference_decode_partial(AttentionArgs a,float* accum,float* lse,float* final_lse) {
  if(a.counts[0]!=0 || a.counts[2]<1 || a.counts[2]>8) return;
  const int row=blockIdx.z/8,head=blockIdx.z%8;
  if(row>=a.counts[2] || a.offsets[row]!=row || a.offsets[row+1]!=row+1) return;
  auto p=ref_parameters(a,nullptr);
  p.b=a.counts[2];p.h=8;p.h_k=8;p.h_h_k_ratio=1;
  p.q_batch_stride=4096;p.q_head_stride=256;p.q_row_stride=128;
  p.seqlen_q=2;p.seqlen_q_rounded=64;p.total_q=p.b*2;
  p.seqlen_k=a.lengths[row];p.seqlen_k_rounded=(p.seqlen_k+127)/128*128;
  p.block_table=const_cast<int*>(a.padded_pages+row*reference_page_slots);p.block_table_batch_stride=0;
  p.oaccum_ptr=accum;p.softmax_lseaccum_ptr=lse;p.softmax_lse_ptr=final_lse;
  p.num_splits=3;p.is_causal=false;
  flash::compute_attn_1rowblock_splitkv<RefTraits,false,false,false,false,true,false,true,false>(p,row,head,0,blockIdx.y,3);
}
__global__ void reference_decode_combine(AttentionArgs a,__nv_bfloat16* out,float* accum,float* lse,float* final_lse) {
  if(a.counts[0]!=0 || a.counts[2]<1 || a.counts[2]>8 || int(blockIdx.x)*4>=a.counts[2]*16) return;
  auto p=ref_parameters(a,out);
  p.b=a.counts[2];p.h=8;p.seqlen_q=2;p.num_splits=3;
  p.o_batch_stride=2048;p.o_head_stride=256;p.o_row_stride=128;
  p.oaccum_ptr=accum;p.softmax_lseaccum_ptr=lse;p.softmax_lse_ptr=final_lse;
  flash::combine_attn_seqk_parallel<RefTraits,4,2,true>(p);
}
static void reference_attention(AttentionArgs a,void* output,void* workspace,CUstream stream) {
  auto s=reinterpret_cast<cudaStream_t>(stream);
  auto y=static_cast<__nv_bfloat16*>(output);
  auto accum=static_cast<float*>(workspace);
  auto lse=accum+3*8*16*128;
  auto final_lse=lse+3*8*16;
  auto table=reinterpret_cast<int*>(final_lse+8*16);
  reference_pad_page_table<<<8,128,0,s>>>(a,table);
  a.padded_pages=table;
  reference_mixed_attention<<<dim3(32,8,16),128,81920,s>>>(a,y);
  reference_decode_partial<<<dim3(1,3,64),128,81920,s>>>(a,accum,lse,final_lse);
  reference_decode_combine<<<32,128,0,s>>>(a,y,accum,lse,final_lse);
  CHECK(cudaGetLastError());
}
static void configure_reference_attention() {
  CHECK(cudaFuncSetAttribute(reference_mixed_attention,cudaFuncAttributeMaxDynamicSharedMemorySize,81920));
  CHECK(cudaFuncSetAttribute(reference_decode_partial,cudaFuncAttributeMaxDynamicSharedMemorySize,81920));
}
#ifndef LF_REFERENCE_STANDALONE
struct AttentionEntry {CUfunction f;int role;};
static thread_local std::vector<AttentionEntry> attention_entries;
static thread_local std::vector<std::pair<const int*,const int*>> attention_offsets;
static thread_local AttentionArgs attention_args;
static thread_local bool attention_pending=false;
static thread_local CUcontext attention_context=nullptr;
static thread_local void *attention_scratch=nullptr,*attention_workspace=nullptr;
static thread_local unsigned long long* attention_checks=nullptr;
static thread_local unsigned long long* attention_records=nullptr;
static thread_local __nv_bfloat16* attention_record_data=nullptr;
static constexpr int record_limit=128,record_history=8256;
static constexpr size_t record_elements=size_t(1+2*record_history)*128;
static thread_local unsigned long long attention_chains=0,attention_original_launches=0;
__global__ void compare_attention(AttentionArgs a,const __nv_bfloat16* expected,const __nv_bfloat16* actual,unsigned long long* checks,unsigned long long* records) {
  unsigned long long seen=0,bad=0,changed=0;float maximum=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.counts[3]*2048;i+=gridDim.x*blockDim.x) {
    float x=__bfloat162float(expected[i]),y=__bfloat162float(actual[i]);
    float d=fabsf(x-y);++seen;
    if(!isfinite(x)||!isfinite(y)||d>0.01f+0.02f*fabsf(x)) {
      ++bad;
      auto slot=atomicAdd(checks+4,1ULL);
      if(slot<record_limit) {
        auto meta=records+slot*8;
        meta[0]=1;meta[1]=i/2048;meta[2]=(i%2048)/128;meta[3]=i%128;
        meta[4]=a.positions[i/2048]+1;meta[5]=__float_as_uint(x);meta[6]=__float_as_uint(y);meta[7]=a.counts[0];
      }
    }
    maximum=fmaxf(maximum,d);if(x!=y) ++changed;
  }
  if(seen) atomicAdd(checks,seen);if(bad) atomicAdd(checks+1,bad);
  atomicMax(checks+2,(unsigned long long)__float_as_uint(maximum));
  if(changed) atomicAdd(checks+3,changed);
}
__global__ void capture_attention_outliers(AttentionArgs a,unsigned long long* records,__nv_bfloat16* data) {
  auto meta=records+blockIdx.x*8;
  if(meta[0]!=1) return;
  int token=int(meta[1]),head=int(meta[2]),history=int(meta[4]);
  if(history>record_history) return;
  int row=0;while(row+1<a.counts[2] && token>=a.offsets[row+1]) ++row;
  auto out=data+blockIdx.x*record_elements;
  for(int i=threadIdx.x;i<128;i+=blockDim.x) out[i]=a.q[size_t(token)*4096+head*128+i];
  for(int i=threadIdx.x;i<history*128;i+=blockDim.x) {
    int pos=i/128,d=i%128,page=a.pages[a.page_offsets[row]+pos/8];
    size_t at=size_t(page)*8192+(pos%8)*1024+(head/2)*128+d;
    out[128+i]=a.k[at];out[128+record_history*128+i]=a.v[at];
  }
  __syncthreads();if(threadIdx.x==0) meta[0]=2;
}
static void attention_register(CUfunction f,const char* name) {
  int role=0;
  if(std::strstr(name,"lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_")==name && !std::strstr(name,"rotary_prepare")) role=1;
  else if(std::strstr(name,"lunaflux_attention_")==name ||
          std::strstr(name,"lunaflux_paged_attention_bf16_")==name) {
    role=std::strstr(name,"merge")?3:((std::strstr(name,"decode") || std::strstr(name,"lunaflux_paged_attention_bf16_")==name)?5:2);
    if(std::strstr(name,"partial")) role=role==5?7:6;
  } else if(std::strstr(name,"lunaflux_luna_dense_projection_bf16_release_v1")==name) role=4;
  if(!role) return;
  for(auto e:attention_entries) if(e.f==f) return;
  attention_entries.push_back({f,role});
  std::fprintf(diagnostic_log,"lf_attention loaded=%s role=%d\n",name,role);std::fflush(diagnostic_log);
  if((LF_SWAP_MASK&4) && !attention_context) {
    CHECK(cuCtxGetCurrent(&attention_context));
    CHECK(cudaMalloc(&attention_scratch,size_t(2048)*2048*2));
    CHECK(cudaMalloc(&attention_workspace,256*1024));
    CHECK(cudaMalloc(&attention_checks,5*sizeof(unsigned long long)));
    CHECK(cudaMemset(attention_checks,0,5*sizeof(unsigned long long)));
    if(LF_SWAP_VERIFY) {
      CHECK(cudaMalloc(&attention_records,record_limit*8*sizeof(unsigned long long)));
      CHECK(cudaMemset(attention_records,0,record_limit*8*sizeof(unsigned long long)));
      CHECK(cudaMalloc(&attention_record_data,record_limit*record_elements*2));
      CHECK(cudaMemset(attention_record_data,0,record_limit*record_elements*2));
    }
    configure_reference_attention();
  }
}
static bool attention_intercept(CUfunction f,CUstream stream,void** args,void** extra) {
  if(!(LF_SWAP_MASK&4)) return false;
  int role=0;for(auto e:attention_entries) if(e.f==f) {role=e.role;break;}
  if(!role) return false;
  if(!args || extra) std::abort();
  auto ptr=[&](int i){return *static_cast<void**>(args[i]);};
  if(role==1) {
    attention_args.offsets=static_cast<int*>(ptr(2));
    attention_offsets.push_back({static_cast<int*>(ptr(0)),attention_args.offsets});
  } else if(role==2 || role==5 || role==6 || role==7) {
    attention_pending=true;++attention_original_launches;
    attention_args.positions=static_cast<int*>(ptr(1));
    attention_args.lengths=static_cast<int*>(ptr(3));
    attention_args.page_offsets=static_cast<int*>(ptr(4));attention_args.pages=static_cast<int*>(ptr(5));
    attention_args.q=static_cast<__nv_bfloat16*>(ptr(6));
    // Partial ABI is Q,K,V,workspace; final-writer ABI is Q,O,K,V.
    const bool partial=role==6 || role==7;
    attention_args.k=static_cast<__nv_bfloat16*>(ptr(partial?7:8));
    attention_args.v=static_cast<__nv_bfloat16*>(ptr(partial?8:9));
    // Decode kernels use raw offsets, matrix prefill uses tile metadata.
    if(role==5 || role==7) {
      attention_args.offsets=static_cast<int*>(ptr(2));
      attention_offsets.push_back({static_cast<int*>(ptr(0)),attention_args.offsets});
    }
    return !LF_SWAP_VERIFY;
  } else if(role==3) {
    attention_args.offsets=static_cast<int*>(ptr(1));
    ++attention_original_launches;return !LF_SWAP_VERIFY;
  } else if(role==4 && attention_pending) {
    attention_args.counts=static_cast<int*>(ptr(0));
    if(!attention_args.offsets) for(auto it=attention_offsets.rbegin();it!=attention_offsets.rend();++it)
      if(it->first==attention_args.counts) {attention_args.offsets=it->second;break;}
    if(!attention_args.offsets || !attention_args.q) {
      std::fprintf(diagnostic_log,"lf_attention failure=missing-descriptor counts=%p offsets=%p q=%p chains=%llu\n",(void*)attention_args.counts,(void*)attention_args.offsets,(void*)attention_args.q,attention_chains);
      std::fflush(diagnostic_log);std::abort();
    }
    auto y=static_cast<__nv_bfloat16*>(ptr(1));
    if(LF_SWAP_VERIFY) {
      CHECK(cudaMemsetAsync(attention_scratch,0xff,size_t(2048)*2048*2,reinterpret_cast<cudaStream_t>(stream)));
      reference_attention(attention_args,attention_scratch,attention_workspace,stream);
      compare_attention<<<256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(attention_args,y,static_cast<__nv_bfloat16*>(attention_scratch),attention_checks,attention_records);
      capture_attention_outliers<<<record_limit,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(attention_args,attention_records,attention_record_data);
      CHECK(cudaGetLastError());
    } else reference_attention(attention_args,y,attention_workspace,stream);
    ++attention_chains;attention_pending=false;attention_args=AttentionArgs{};
  }
  return false;
}
static void attention_release(CUcontext context) {
  if(context!=attention_context) return;
  CHECK(cuCtxSetCurrent(context));CHECK(cudaDeviceSynchronize());
  unsigned long long c[5]={};CHECK(cudaMemcpy(c,attention_checks,sizeof(c),cudaMemcpyDeviceToHost));
  unsigned bits=unsigned(c[2]);float maximum;std::memcpy(&maximum,&bits,4);
  std::fprintf(diagnostic_log,"lf_attention chains=%llu intercepted=%llu checked=%llu violations=%llu max_abs=%g nonbitwise=%llu pending=%d\n",attention_chains,attention_original_launches,c[0],c[1],maximum,c[3],int(attention_pending));
  std::fflush(diagnostic_log);
  if(LF_SWAP_VERIFY && c[4]) {
    char path[4096];std::snprintf(path,sizeof(path),"%s/outliers-%ld.bin",LF_SWAP_LOG_DIRECTORY,long(getpid()));
    FILE* f=std::fopen(path,"wx");if(!f) std::abort();
    unsigned long long count=c[4]<record_limit?c[4]:record_limit;
    std::fwrite(&count,sizeof(count),1,f);
    std::vector<unsigned long long> meta(count*8);
    CHECK(cudaMemcpy(meta.data(),attention_records,meta.size()*8,cudaMemcpyDeviceToHost));
    std::fwrite(meta.data(),8,meta.size(),f);
    std::vector<__nv_bfloat16> data(record_elements);
    for(size_t i=0;i<count;++i) {
      CHECK(cudaMemcpy(data.data(),attention_record_data+i*record_elements,record_elements*2,cudaMemcpyDeviceToHost));
      std::fwrite(data.data(),2,record_elements,f);
    }
    if(std::fclose(f)) std::abort();
    std::fprintf(diagnostic_log,"lf_attention outliers_recorded=%llu total=%llu path=%s\n",count,c[4],path);std::fflush(diagnostic_log);
  }
  if(attention_records) CHECK(cudaFree(attention_records));
  if(attention_record_data) CHECK(cudaFree(attention_record_data));
  CHECK(cudaFree(attention_scratch));CHECK(cudaFree(attention_workspace));CHECK(cudaFree(attention_checks));
  attention_context=nullptr;attention_entries.clear();
  if(c[1] || attention_pending || !attention_chains) std::abort();
}
#endif
