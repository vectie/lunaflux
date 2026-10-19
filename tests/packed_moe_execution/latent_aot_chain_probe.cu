// Offline replay of the exact serving cubins, not a substitute implementation.
// The one-hot bank makes both cache representations the same mathematical
// problem. History preparation is outside timing; projection/attention launches
// are timed individually. Indexing, history copy and the rest of GLM are absent.
#include <cuda.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void ck(CUresult r) {
  if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r,&s); fprintf(stderr,"%s\n",s); exit(2); }
}
using B=__nv_bfloat16;
struct Buffer {
  CUdeviceptr p; size_t bytes;
  explicit Buffer(size_t n):bytes(n){ck(cuMemAlloc(&p,n));ck(cuMemsetD8(p,0,n));}
  ~Buffer(){ck(cuMemFree(p));}
  template<class T> void put(const std::vector<T>&v){if(v.size()*sizeof(T)>bytes)exit(3);ck(cuMemcpyHtoD(p,v.data(),v.size()*sizeof(T)));}
  std::vector<B> get(){std::vector<B> v(bytes/2);ck(cuMemcpyDtoH(v.data(),p,bytes));return v;}
};
struct Entry { CUfunction f; int grid,block; };
static Entry entry(CUmodule m,const char *name,int grid,int block){Entry e{{},grid,block};ck(cuModuleGetFunction(&e.f,m,name));return e;}
static void run(Entry e,std::vector<void*> args){ck(cuLaunchKernel(e.f,e.grid,1,1,e.block,1,1,0,nullptr,args.data(),nullptr));}
static B value(int key,int component){return __float2bfloat16_rn(float((key*13+component*7)%127-63)/256.f);}
static unsigned bits(B x){unsigned short v;memcpy(&v,&x,2);return v;}
static float median(std::vector<float> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char **argv){
  if(argc!=4)return 1;
  const bool once=strcmp(argv[3],"once")==0;
  ck(cuInit(0));CUdevice dev;ck(cuDeviceGet(&dev,0));CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev));ck(cuCtxSetCurrent(ctx));
  CUmodule old,next;ck(cuModuleLoad(&old,argv[1]));ck(cuModuleLoad(&next,argv[2]));
  auto project=entry(old,"glm_sparse_3_projection_kv_b",1024,128);
  auto split=entry(old,"glm_sparse_3_projection_kv_split",4096,256);
  auto oa=entry(old,"glm_sparse_3_attention_attention_retained_scores_v1",2048,256);
  auto absorb=entry(next,"glm_sparse_3_projection_q_absorb",1024,128);
  auto na=entry(next,"glm_sparse_3_attention_attention_retained_scores_v1",2048,256);
  auto expand=entry(next,"glm_sparse_3_value_expand",512,128);
  {
    constexpr int R=64,H=8192,N=32,D=256,K=512,W=2051;
    Buffer counts(20),positions(R*4),key_count(4),valid(8208*4),seq(R*4),offsets(9*4),pages(4096*4),gens(4096*4),selected(R*W*4);
    Buffer q(R*N*D*2),weight(N*(D+D)*K*2),latent(H*K*2),current(R*K*2),packed(R*N*(D+D)*2),current_k(R*N*D*2),current_v(R*N*D*2);
    Buffer ek(H*N*D*2),ev(H*N*D*2),aq(R*N*K*2),context(R*N*K*2),old_out(R*N*D*2),new_out(R*N*D*2),os(R*N*H*4),ns(R*N*H*4);
    counts.put(std::vector<int>{1,0,1,R,0});key_count.put(std::vector<int>{H});valid.put(std::vector<int>(8208,1));
    offsets.put(std::vector<int>{0,512,512,512,512,512,512,512,512});
    std::vector<int> page(4096),generation(4096,1);for(int i=0;i<512;++i)page[i]=i;pages.put(page);gens.put(generation);
    std::vector<B> bank(N*(D+D)*K,__float2bfloat16_rn(0));
    for(int h=0;h<N;++h)for(int c=0;c<D;++c){bank[(h*2*D+c)*K+c]=__float2bfloat16_rn(1);bank[(h*2*D+D+c)*K+D+c]=__float2bfloat16_rn(1);}weight.put(bank);
    std::vector<B> queries(R*N*D);for(size_t i=0;i<queries.size();++i)queries[i]=value(i/D,i%D);q.put(queries);
    std::vector<B> l(H*K);for(int k=0;k<H;++k)for(int c=0;c<K;++c)l[k*K+c]=value(k,c);latent.put(l);
    std::vector<B> expanded(H*N*D);
    for(int k=0;k<H;++k)for(int h=0;h<N;++h)for(int c=0;c<D;++c)expanded[(k*N+h)*D+c]=l[k*K+c];ek.put(expanded);
    for(int k=0;k<H;++k)for(int h=0;h<N;++h)for(int c=0;c<D;++c)expanded[(k*N+h)*D+c]=l[k*K+D+c];ev.put(expanded);
    auto old_chain=[&](int stage){
      if(stage==0)run(project,{&counts.p,&current.p,&weight.p,&packed.p});
      if(stage==1)run(split,{&counts.p,&packed.p,&current_k.p,&current_v.p});
      if(stage==2)run(oa,{&counts.p,&positions.p,&key_count.p,&valid.p,&q.p,&ek.p,&ev.p,&selected.p,&old_out.p,&os.p,&seq.p,&offsets.p,&pages.p,&gens.p});
    };
    auto new_chain=[&](int stage){
      if(stage==0)run(absorb,{&counts.p,&q.p,&weight.p,&aq.p});
      if(stage==1)run(na,{&counts.p,&positions.p,&key_count.p,&valid.p,&aq.p,&latent.p,&latent.p,&selected.p,&context.p,&ns.p,&seq.p,&offsets.p,&pages.p,&gens.p});
      if(stage==2)run(expand,{&counts.p,&context.p,&weight.p,&new_out.p});
    };
    for(int history: (once?std::vector<int>{8192}:std::vector<int>{2048,8192}))for(int stride: (once?std::vector<int>{4}:std::vector<int>{1,4})){
      current.put(std::vector<B>(l.begin()+(history-R)*K,l.begin()+history*K));
      std::vector<int> pos(R),ids(R*W,-1);
      for(int r=0;r<R;++r){pos[r]=history-R+r;for(int j=0;j<2048&&j*stride<=pos[r];++j)ids[r*W+j]=j*stride;}
      positions.put(pos);selected.put(ids);
      std::vector<float> times[6];CUevent start,end;ck(cuEventCreate(&start,0));ck(cuEventCreate(&end,0));
      for(int trial=0;trial<(once?1:11);++trial){
        for(int turn=0;turn<2;++turn){const bool is_new=(turn+(trial&1))%2;
          for(int s=0;s<3;++s){ck(cuEventRecord(start,nullptr));if(is_new)new_chain(s);else old_chain(s);ck(cuEventRecord(end,nullptr));ck(cuEventSynchronize(end));float ms;ck(cuEventElapsedTime(&ms,start,end));if(once||trial>=2)times[(is_new?3:0)+s].push_back(ms);}
        }
      }
      ck(cuEventDestroy(start));ck(cuEventDestroy(end));
      auto a=old_out.get(),b=new_out.get();size_t differences=0;for(size_t i=0;i<a.size();++i)differences+=bits(a[i])!=bits(b[i]);
      printf("history=%d stride=%d exact=%zu/%zu old_project_ms=%.6f old_split_ms=%.6f old_attention_ms=%.6f new_absorb_ms=%.6f new_attention_ms=%.6f new_expand_ms=%.6f\n",history,stride,a.size()-differences,a.size(),median(times[0]),median(times[1]),median(times[2]),median(times[3]),median(times[4]),median(times[5]));
      if(differences){fprintf(stderr,"one-hot equivalent fixture mismatch\n");return 4;}
    }
  }
  ck(cuModuleUnload(next));ck(cuModuleUnload(old));ck(cuDevicePrimaryCtxRelease(dev));puts("released=true");
}
