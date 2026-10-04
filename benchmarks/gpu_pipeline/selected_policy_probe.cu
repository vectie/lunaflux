// Offline AOT schedule calibration. Geometry comes from an explicit workload
// spec; no model name, runtime code or production CUDA ABI is linked here.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "selected_policy_geometry.h"
#define CK(x) do { auto e=(x); if(e!=0){std::fprintf(stderr,"CUDA %d line %d\n",int(e),__LINE__);std::exit(2);} }while(0)
using Fields=std::map<std::string,std::string>;
static Fields fields(const std::string& path) {
  std::ifstream file(path); if(!file) {std::fprintf(stderr,"missing %s\n",path.c_str());std::exit(1);}
  Fields out; std::string line;
  while(std::getline(file,line)){auto p=line.find('=');if(p!=std::string::npos)out.emplace(line.substr(0,p),line.substr(p+1));}
  return out;
}
static int number(const Fields& f,const char* key){return std::stoi(f.at(key));}
static std::vector<int> tuple(std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream in(s);std::vector<int> out;int n;while(in>>n)out.push_back(n);return out;}
static bool split_decode_law(const std::string& law) {
  for(const char* supported:{"blockwise-f32-probability-v1","dual-score-blockwise-f32-probability-v2",
      "blockwise-fma-f32-probability-v3","owned2-blockwise-fma-f32-probability-v4",
      "owned4-blockwise-f32-probability-v4","owned4-blockwise-fma-f32-probability-v4",
      "owned8-blockwise-f32-probability-v4","owned8-blockwise-fma-f32-probability-v4",
      "grouped-head-matrix-bf16-probability-v1"})if(law==supported)return true;
  return false;
}
struct Buffer {
  void* p=nullptr; size_t bytes;
  explicit Buffer(size_t n):bytes(n){CK(cudaMalloc(&p,n));clear();}
  ~Buffer(){CK(cudaFree(p));}
  Buffer(const Buffer&)=delete;
  void clear(){CK(cudaMemset(p,0,bytes));}
  template<class T> void put(const std::vector<T>& v){if(v.size()*sizeof(T)>bytes)std::exit(3);CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<__nv_bfloat16> read(){std::vector<__nv_bfloat16> v(bytes/2);CK(cudaMemcpy(v.data(),p,bytes,cudaMemcpyDeviceToHost));return v;}
};
struct Kernel {
  CUmodule module; CUfunction function,merge=nullptr,rotary_prepare=nullptr;
  unsigned gx,gy,gz,b,shared,merge_y=0,rotary_grid=0,rotary_block=0;
  int regs,resident,local; std::string law; size_t rotary_bytes=0;
  void* workspace=nullptr;
  explicit Kernel(const std::string& root, int bucket_tokens, int profile_rows, bool ingress, bool decode, bool postprocess=false, bool partitioned=false){
    auto r=fields(root+"/kernel.recipe");
    if(partitioned && !r.count("merge_function_symbol")) {
      if(!decode || !r.count("partition_function_symbol") || !r.count("partition_merge_function_symbol"))std::exit(1);
      r["function_symbol"]=r.at("partition_function_symbol");
      r["merge_function_symbol"]=r.at("partition_merge_function_symbol");
      r["grid"]=r.at("partition_grid");r["merge_grid"]=r.at("partition_merge_grid");
      r["block"]=r.at("partition_block");r["shared_memory_bytes"]=r.at("partition_shared_memory_bytes");
      r["workspace_bytes"]=r.at("partition_workspace_bytes");r["numeric_law"]=r.at("partition_numeric_law");
    }
    auto grid=tuple(r.at("grid")),block=tuple(r.at("block"));
    gx=grid.at(0);gy=grid.at(1);gz=grid.at(2);b=block.at(0);shared=number(r,"shared_memory_bytes");
    const int tile=(decode||postprocess) ? 1 : number(r,"query_tile_rows");
    const bool metadata=!ingress&&!decode&&!postprocess&&number(r,"query_metadata_version")==1;
    gx=postprocess?std::min(gx,unsigned(bucket_tokens)):selected_grid_x(gx,bucket_tokens,profile_rows,tile,metadata,decode);
    law=r.count("numeric_law") ? r.at("numeric_law") : "legacy-declared-law";
    CK(cuModuleLoad(&module,(root+"/kernel.cubin").c_str()));
    CK(cuModuleGetFunction(&function,module,r.at(postprocess?"symbol":"function_symbol").c_str()));
    if(r.count("rotary_cache_policy")) {
      if(!ingress || r.at("rotary_cache_policy")!="step-prepared-f32-pairs-v1")std::exit(1);
      CK(cuModuleGetFunction(&rotary_prepare,module,r.at("rotary_prepare_symbol").c_str()));
      rotary_grid=tuple(r.at("rotary_prepare_grid")).at(0);
      rotary_block=tuple(r.at("rotary_prepare_block")).at(0);
      rotary_bytes=std::stoull(r.at("rotary_cache_bytes"));
      if(rotary_grid==0 || rotary_block==0 || rotary_block>1024)std::exit(1);
    }
    if(r.count("merge_function_symbol")) {
      if(!decode || gz<=1 || !split_decode_law(law))std::exit(1);
      CK(cuModuleGetFunction(&merge,module,r.at("merge_function_symbol").c_str()));
      auto bytes=std::stoull(r.at("workspace_bytes"));
      if(bytes==0 || bytes>16777216)std::exit(1);
      CK(cudaMalloc(&workspace,bytes));CK(cudaMemset(workspace,0,bytes));
      merge_y=tuple(r.at("merge_grid")).at(1);
    }
    if(shared)CK(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
    CK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
    CK(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
    CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,function,b,shared));
  }
  ~Kernel(){if(workspace)CK(cudaFree(workspace));CK(cuModuleUnload(module));}
  void prepare(void* counts,void* positions,void* cache,size_t cache_bytes) {
    if(!rotary_prepare)return;
    if(cache_bytes<rotary_bytes)std::exit(1);
    void* args[]={&counts,&positions,&cache};
    CK(cuLaunchKernel(rotary_prepare,rotary_grid,1,1,rotary_block,1,1,0,nullptr,args,nullptr));
  }
  void launch(void** a){
    if(merge){
      void* partial[]={a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[8],a[9],&workspace};
      void* combine[]={a[0],a[2],&workspace,a[7]};
      CK(cuLaunchKernel(function,gx,gy,gz,b,1,1,shared,nullptr,partial,nullptr));
      CK(cuLaunchKernel(merge,gx,merge_y,1,b,1,1,0,nullptr,combine,nullptr));
    }else{CK(cuLaunchKernel(function,gx,gy,1,b,1,1,shared,nullptr,a,nullptr));}
  }
  double time(void** a,const std::vector<std::unique_ptr<Buffer>>& weights={}){
    auto repeat=[&](int i){
      if(weights.empty()){launch(a);return;}
      // Immutable, independently allocated layer operands. Pointer selection
      // is outside the device work; no uploads or cache flushes are timed.
      void* selected[16];std::copy(a,a+16,selected);
      const int layer=i%28;
      selected[7]=&weights[layer*3]->p;
      selected[8]=&weights[layer*3+1]->p;
      selected[9]=&weights[layer*3+2]->p;
      launch(selected);
    };
    for(int i=0;i<(weights.empty()?3:28);i++)repeat(i);CK(cudaDeviceSynchronize());
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    CK(cudaEventRecord(start));for(int i=0;i<30;i++)repeat(i);
    CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));return ms*1000.0/30;
  }
};
static std::vector<__nv_bfloat16> values(size_t n,int salt){
  std::vector<__nv_bfloat16> v(n);
  for(size_t i=0;i<n;i++)v[i]=__float2bfloat16_rn(float(int((i*17+salt)%61)-30)/64.f);
  return v;
}
static bool same(const std::vector<__nv_bfloat16>& a,const std::vector<__nv_bfloat16>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*2);}
int main(int argc,char**argv){
  // SPEC BASELINE_DIRECTORY CANDIDATE_DIRECTORY TOKENS ROWS HISTORY
  if(argc<7 || argc>12)return 1;
  bool check_only=false,mixed=false,decode_envelope=false,partitioned=false,streaming=false;
  for(int i=7;i<argc;i++){
    if(std::string(argv[i])=="--check-only" && !check_only)check_only=true;
    else if(std::string(argv[i])=="--mixed" && !mixed)mixed=true;
    else if(std::string(argv[i])=="--decode-envelope" && !decode_envelope)decode_envelope=true;
    else if(std::string(argv[i])=="--decode-partitioned" && !partitioned)partitioned=true;
    else if(std::string(argv[i])=="--streaming-ingress-weights" && !streaming)streaming=true;
    else return 1;
  }
  auto spec=fields(argv[1]);const auto kind=spec.at("kind");
  const bool ingress=kind=="ingress",decode=kind=="decode",postprocess=kind=="postprocess";
  if(streaming&&!ingress)return 1;
  const int tokens=std::stoi(argv[4]),rows=std::stoi(argv[5]),past=std::stoi(argv[6]);
  const int qh=number(spec,"query_heads"),kh=number(spec,"key_value_heads"),d=number(spec,"head_dimension");
  const int input_width=number(spec,"input_width"),page=number(spec,"tokens_per_page"),stride=number(spec,"page_stride_values");
  const int max_rows=number(spec,"maximum_rows"),max_tokens=number(spec,"maximum_tokens"),max_pages=number(spec,"maximum_pages");
  // History is limited by the declared page arena, not an old 8K experiment.
  // Bound before host/device allocation, including uneven per-request tails.
  if(tokens<rows||rows<1||rows>max_rows||tokens>max_tokens||page<1||max_pages<1||past<0||
     (decode&&tokens!=rows)||(mixed&&(decode||rows<2||tokens<=rows)))return 1;
  const int64_t largest_row=mixed?(int64_t(tokens)-1+rows-2)/(rows-1):(int64_t(tokens)+rows-1)/rows;
  if(int64_t(past)+largest_row>int64_t(page)*max_pages)return 1;
  if((decode_envelope || partitioned) && !decode)return 1;
  // Explicit replay of a traced capacity-grid graph, including inactive rows.
  // Do not silently treat a compact synthetic bucket as the serving envelope.
  const int bucket_tokens=decode_envelope?max_rows:selected_bucket_tokens(tokens,decode?max_rows:max_tokens);
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  Kernel baseline(argv[2],bucket_tokens,max_rows,ingress,decode,postprocess,partitioned),candidate(argv[3],bucket_tokens,max_rows,ingress,decode,postprocess,partitioned);
  std::printf("geometry mode=%s tokens=%d rows=%d history=%d bucket_tokens=%d bucket_rows=%d old_grid=%u,%u,%u new_grid=%u,%u,%u numeric_law=%s\n",
    decode_envelope?"traced-decode-envelope":"runtime-bucket",tokens,rows,past,bucket_tokens,std::min(max_rows,bucket_tokens),baseline.gx,baseline.gy,baseline.gz,candidate.gx,candidate.gy,candidate.gz,candidate.law.c_str());
  std::vector<int> offsets{0},lengths,pages,po{0},positions;
  for(int r=0;r<rows;r++){
    const int prefill_rows=mixed?rows-1:rows,prefill_tokens=mixed?tokens-1:tokens;
    int n=mixed&&r==rows-1?1:prefill_tokens/prefill_rows+(r<prefill_tokens%prefill_rows),length=past+n;
    lengths.push_back(length);for(int t=0;t<n;t++)positions.push_back(past+t);
    offsets.push_back(offsets.back()+n);
    for(int p=0;p<(length+page-1)/page;p++)pages.push_back(int(pages.size()));
    po.push_back(int(pages.size()));
  }
  if(int(pages.size())>max_pages)return 1;
  // A bounded bijection breaks the logical-page == physical-page shortcut.
  // Reverse and rotate is legal for every page count, unlike an unchecked
  // odd-multiplier hash whose coprimality depends on workload shape.
  const int physical_pages=int(pages.size());
  for(int i=0;i<physical_pages;i++)pages[i]=(physical_pages-1-i+physical_pages/3)%physical_pages;
  std::vector<int> counts{decode?0:(mixed?rows-1:rows),decode?rows:(mixed?1:0),rows,tokens,int(pages.size())};
  // Same bounded CSR contract as luna_attention_metadata, prepared off timer.
  std::vector<int> metadata(4,0);
  for(int bucket=0;bucket<4;bucket++){
    int width=16<<bucket,capacity=max_rows+(max_tokens-max_rows)/width,base=metadata.size(),count=0;
    metadata.resize(base+capacity*8);
    for(int r=0;r<rows;r++)for(int t=offsets[r];t<offsets[r+1];t+=width){
      int record[8]={r,t,offsets[r],offsets[r+1],lengths[r],po[r],po[r+1],positions[std::min(t+width,offsets[r+1])-1]+1};
      if(count>=capacity)return 1;std::copy(record,record+8,metadata.begin()+base+count*8);count++;
    }
    metadata[bucket]=count;
  }
  auto x=values(size_t(tokens)*input_width,3);
  auto key=values(size_t(pages.size())*stride,29),value=values(key.size(),31);
  if(!ingress&&!postprocess){
    // Dense-current and paged-history views describe the same immutable KV.
    for(int r=0;r<rows;r++)for(int t=offsets[r];t<offsets[r+1];t++)for(int h=0;h<kh;h++)for(int c=0;c<d;c++){
      size_t address=size_t(pages[po[r]+positions[t]/page])*stride+(positions[t]%page*kh+h)*d+c;
      key[address]=x[size_t(t)*input_width+qh*d+h*d+c];
      value[address]=x[size_t(t)*input_width+(qh+kh)*d+h*d+c];
    }
  }
  Buffer dc(20),dp(positions.size()*4),doff(offsets.size()*4),dl(lengths.size()*4),dpo(po.size()*4),dpi(pages.size()*4),dm(metadata.size()*4);
  dc.put(counts);dp.put(positions);doff.put(offsets);dl.put(lengths);dpo.put(po);dpi.put(pages);dm.put(metadata);
  Buffer dx(x.size()*2),dk(key.size()*2),dv(value.size()*2),out(size_t(tokens)*((ingress||postprocess)?(qh+2*kh)*d:qh*d)*2);
  dx.put(x);dk.put(key);dv.put(value);
  Buffer qw(size_t(qh)*d*(ingress?input_width:1)*2),kw(size_t(kh)*d*(ingress?input_width:1)*2),vw(kw.bytes),norm(d*2);
  Buffer rotary(ingress?size_t(max_tokens)*(d/2)*sizeof(float2):sizeof(float2));
  qw.put(values(qw.bytes/2,7));kw.put(values(kw.bytes/2,11));vw.put(values(vw.bytes/2,13));
  norm.put(std::vector<__nv_bfloat16>(d,__float2bfloat16_rn(1)));
  void* row_data=(!ingress&&!decode&&!postprocess&&number(spec,"query_metadata_version")==1)?dm.p:doff.p;
  void* attention_args[]={&dc.p,&dp.p,&row_data,&dl.p,&dpo.p,&dpi.p,&dx.p,&out.p,&dk.p,&dv.p};
  void* ingress_args[]={&dc.p,&dp.p,&doff.p,&dl.p,&dpo.p,&dpi.p,&dx.p,&qw.p,&kw.p,&vw.p,&norm.p,&norm.p,&out.p,&dk.p,&dv.p,&rotary.p};
  void* postprocess_args[]={&dc.p,&dp.p,&doff.p,&dl.p,&dpo.p,&dpi.p,&dx.p,&norm.p,&norm.p,&out.p,&dk.p,&dv.p};
  void** args=ingress?ingress_args:(postprocess?postprocess_args:attention_args);
  baseline.prepare(dc.p,dp.p,rotary.p,rotary.bytes);
  baseline.launch(args);CK(cudaDeviceSynchronize());auto expected=out.read(),expected_key=dk.read(),expected_value=dv.read();
  out.clear();dk.put(key);dv.put(value);
  candidate.prepare(dc.p,dp.p,rotary.p,rotary.bytes);
  candidate.launch(args);CK(cudaDeviceSynchronize());auto actual=out.read();
  double error=0;bool nonzero=false;
  for(size_t i=0;i<actual.size();i++){
    float a=__bfloat162float(actual[i]),b=__bfloat162float(expected[i]);
    if(!std::isfinite(a)||!std::isfinite(b))return 3;
    error=std::max(error,double(std::abs(a-b)));nonzero|=a!=0;
  }
  bool bitwise=same(expected,actual);
  if(!nonzero||(ingress&&!bitwise)||error>0.003||!same(expected_key,dk.read())||!same(expected_value,dv.read())){
    std::fprintf(stderr,"differential failed kind=%s maxabs=%g bitwise=%d\n",kind.c_str(),error,int(bitwise));return 3;
  }
  // Independent sampled attention reference; uses full FP32 probabilities and
  // FP64 accumulation, not the compiler's recurrence or fragment mapping.
  double oracle_error=0;
  if(postprocess){
    const double epsilon=std::stod(spec.at("norm_epsilon")),theta=std::stod(spec.at("rope_theta"));
    const auto actual_key=dk.read(),actual_value=dv.read();
    for(int r=0;r<rows;r++)for(int t=offsets[r];t<offsets[r+1];t++)for(int h=0;h<qh+2*kh;h++){
      std::vector<double> head(d);double energy=0;
      for(int c=0;c<d;c++){head[c]=__bfloat162float(x[size_t(t)*input_width+h*d+c]);energy+=head[c]*head[c];}
      if(h<qh+kh){
        const double inverse=1/std::sqrt(energy/d+epsilon);
        for(auto& scalar:head)scalar=__bfloat162float(__float2bfloat16_rn(float(scalar*inverse)));
        for(int c=0;c<d/2;c++){
          const double angle=positions[t]*std::pow(theta,-2.0*c/d),a=head[c],b=head[c+d/2];
          head[c]=__bfloat162float(__float2bfloat16_rn(float(a*std::cos(angle)-b*std::sin(angle))));
          head[c+d/2]=__bfloat162float(__float2bfloat16_rn(float(b*std::cos(angle)+a*std::sin(angle))));
        }
      }
      for(int c=0;c<d;c++){
        oracle_error=std::max(oracle_error,std::abs(double(__bfloat162float(actual[size_t(t)*input_width+h*d+c]))-head[c]));
        if(h>=qh){
          int kvh=h<qh+kh?h-qh:h-qh-kh;
          size_t address=size_t(pages[po[r]+positions[t]/page])*stride+(positions[t]%page*kh+kvh)*d+c;
          oracle_error=std::max(oracle_error,std::abs(double(__bfloat162float((h<qh+kh?actual_key:actual_value)[address]))-head[c]));
        }
      }
    }
    if(oracle_error>0.015){std::fprintf(stderr,"postprocess oracle failed error=%g\n",oracle_error);return 4;}
  }
  if(!ingress&&!postprocess)for(int r=0;r<rows;r++)for(int t:{offsets[r],(offsets[r]+offsets[r+1]-1)/2,offsets[r+1]-1})for(int h:{0,qh-1}){
    int length=positions[t]+1;std::vector<double> scores(length);double maximum=-INFINITY;
    for(int k=0;k<length;k++){
      size_t address=size_t(pages[po[r]+k/page])*stride+(k%page*kh+h/(qh/kh))*d;
      double dot=0;for(int c=0;c<d;c++)dot+=double(__bfloat162float(x[size_t(t)*input_width+h*d+c]))*__bfloat162float(key[address+c]);
      scores[k]=dot/std::sqrt(double(d));maximum=std::max(maximum,scores[k]);
    }
    double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
    for(int c=0;c<d;c++){
      double sum=0;for(int k=0;k<length;k++)sum+=scores[k]*__bfloat162float(value[size_t(pages[po[r]+k/page])*stride+(k%page*kh+h/(qh/kh))*d+c]);
      oracle_error=std::max(oracle_error,std::abs(double(__bfloat162float(actual[size_t(t)*qh*d+h*d+c]))-sum/denominator));
    }
    if(oracle_error>0.003){std::fprintf(stderr,"oracle failed error=%g\n",oracle_error);return 4;}
  }
  std::printf("resources registers=%d resident_blocks=%d local_bytes=%d\n",candidate.regs,candidate.resident,candidate.local);
  if(candidate.rotary_prepare) {
    // Kernel timing below measures the consumer. Preparation is once per whole
    // decoder step, not once per layer; report it independently rather than
    // silently counting a free cache or charging every layer another launch.
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    CK(cudaEventRecord(start));
    for(int repeat=0;repeat<100;repeat++)candidate.prepare(dc.p,dp.p,rotary.p,rotary.bytes);
    CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;
    CK(cudaEventElapsedTime(&ms,start,end));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    std::printf("rotary_cache preparation_us=%.6f bytes=%zu consumer_timing_excludes_preparation=true\n",ms*10.0,rotary.bytes);
  }
  if(check_only){std::printf("correctness=passed bitwise=%s maxabs=%g oracle_maxabs=%g\n",bitwise?"true":"false",error,oracle_error);return 0;}
  std::vector<std::unique_ptr<Buffer>> layer_weights;
  if(streaming){
    size_t total=28*(qw.bytes+kw.bytes+vw.bytes);
    if(total>1024ULL*1024*1024)return 1;
    for(int layer=0;layer<28;layer++)for(int operand=0;operand<3;operand++){
      size_t bytes=operand==0?qw.bytes:kw.bytes;
      auto weight=std::make_unique<Buffer>(bytes);
      weight->put(values(bytes/2,operand==0?7:(operand==1?11:13)));
      layer_weights.push_back(std::move(weight));
    }
    std::printf("working_set mode=distinct-layer-weights layers=28 bytes=%zu identical_values=true\n",total);
    // Validate every rotated allocation, not just the original oracle pair.
    for(int layer=0;layer<28;layer++){
      void* selected[16];std::copy(args,args+16,selected);
      for(int operand=0;operand<3;operand++)selected[7+operand]=&layer_weights[layer*3+operand]->p;
      candidate.launch(selected);CK(cudaDeviceSynchronize());
      if(!same(expected,out.read())||!same(expected_key,dk.read())||!same(expected_value,dv.read()))return 3;
    }
  }
  for(int trial=0;trial<5;trial++){
    double a,b;if(trial%2){b=candidate.time(args,layer_weights);a=baseline.time(args,layer_weights);}else{a=baseline.time(args,layer_weights);b=candidate.time(args,layer_weights);}
    std::printf("tokens=%d rows=%d history=%d trial=%d old_us=%.6f new_us=%.6f bitwise=%s maxabs=%g oracle_maxabs=%g\n",tokens,rows,past,trial,a,b,bitwise?"true":"false",error,oracle_error);
  }
}
