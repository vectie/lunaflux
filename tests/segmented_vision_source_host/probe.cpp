#include "cuda_bf16.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <algorithm>
extern "C" void vision_patch_rows(const int32_t*,const __nv_bfloat16*,__nv_bfloat16*);
extern "C" void vision_position(const int32_t*,const int32_t*,const __nv_bfloat16*,__nv_bfloat16*);
extern "C" void vision_early_gather(const int32_t*,const int32_t*,const __nv_bfloat16*,__nv_bfloat16*);
extern "C" void vision_early_mean_residual(const int32_t*,const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*);
extern "C" void vision_raw_attention(const int32_t*,const int32_t*,const int32_t*,const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,__nv_bfloat16*);
#ifdef LUNAFLUX_TEST_IMAGE_SCATTER
extern "C" void lunaflux_minicpm_v47_image_scatter(const int32_t*,const int32_t*,const int32_t*,const int32_t*,const uint16_t*,uint16_t*);
#endif
static void require(bool condition,const char* message) {
  if (!condition) { std::fprintf(stderr,"vision host source probe: %s\n",message); std::exit(1); }
}
template<class F> static void elements(int count,F run) {
  blockDim={256,1,1};
  for(int i=0;i<count;++i) { blockIdx={i/256,0,0};threadIdx={i%256,0,0};run(); }
}
template<class F> static void attention(int row,int head,F run) {
  std::vector<std::thread> threads;
  for(int lane=0;lane<128;++lane) threads.emplace_back([=]{
    blockIdx={row,head,0};threadIdx={lane,0,0};blockDim={128,1,1};run();
  });
  for(auto& thread:threads) thread.join();
}
int main() {
  using B=__nv_bfloat16;
  int32_t counts[4]={0,0,0,4};
  std::vector<B> pixels(4*588),patches(4*588);
  for(int i=0;i<int(pixels.size());++i) pixels[i].bits=uint16_t(i*17+1);
  elements(int(patches.size()),[&]{vision_patch_rows(counts,pixels.data(),patches.data());});
  for(int row=0;row<4;++row) for(int c=0;c<3;++c) for(int py=0;py<14;++py) for(int px=0;px<14;++px) {
    int src=(c*14+py)*4*14+row*14+px;
    int dst=row*588+c*196+py*14+px;
    require(patches[dst].bits==pixels[src].bits,"NaViT patch gather changed a word");
  }
  std::vector<B> raster(4*1152),permuted(4*1152);
  for(int i=0;i<int(raster.size());++i) raster[i].bits=uint16_t(i*31+7);
  int32_t indices[4]={2,0,3,1};
  elements(int(raster.size()),[&]{vision_early_gather(counts,indices,raster.data(),permuted.data());});
  for(int row=0;row<4;++row) for(int c=0;c<1152;++c)
    require(permuted[row*1152+c].bits==raster[indices[row]*1152+c].bits,"window permutation");
  for(int row=0;row<4;++row) for(int c=0;c<1152;++c) raster[row*1152+c]=__float2bfloat16_rn(float(row+1));
  std::vector<B> branch(1152,__float2bfloat16_rn(0.5f)),merged(1152);
  elements(1152,[&]{vision_early_mean_residual(counts,raster.data(),branch.data(),merged.data());});
  for(auto value:merged) require(__bfloat162float(value)==3.0f,"mean residual BF16 stages");
  std::vector<B> positions(4*1152);
  for(int row=0;row<4;++row) for(int c=0;c<1152;++c) positions[row*1152+c]=__float2bfloat16_rn(row*0.5f);
  int32_t ids[4]={3,0,2,1};
  elements(int(raster.size()),[&]{vision_position(counts,ids,positions.data(),raster.data());});
  for(int row=0;row<4;++row) for(int c=0;c<1152;++c)
    require(__bfloat162float(raster[row*1152+c])==float(row+1)+ids[row]*0.5f,"position add");
  counts[3]=3;
  std::vector<B> q(3*1152),k(3*1152),v(3*1152),out(3*1152);
  for(int row=0;row<3;++row) for(int c=0;c<1152;++c) v[row*1152+c]=__float2bfloat16_rn(row==0?1.0f:2.0f);
  int32_t starts[3]={0,0,0},ends[3]={3,3,3};
  for(int row=0;row<3;++row) attention(row,0,[&]{vision_raw_attention(counts,starts,ends,q.data(),k.data(),v.data(),out.data());});
  for(int row=0;row<3;++row) for(int c=0;c<72;++c)
    require(__bfloat162float(out[row*1152+c])==1.671875f,"normalized BF16 probability stage");
  counts[3]=4;
  q.resize(4*1152);k.resize(4*1152);v.resize(4*1152);out.resize(4*1152);
  const float values[4]={1,3,10,20};
  for(int row=0;row<4;++row) for(int c=0;c<1152;++c) v[row*1152+c]=__float2bfloat16_rn(values[row]);
  int32_t isolated_starts[4]={0,0,2,2},isolated_ends[4]={2,2,4,4};
  for(int row=0;row<4;++row) attention(row,0,[&]{vision_raw_attention(counts,isolated_starts,isolated_ends,q.data(),k.data(),v.data(),out.data());});
  for(int row=0;row<4;++row) for(int c=0;c<72;++c)
    require(__bfloat162float(out[row*1152+c])==(row<2?2.0f:15.0f),"attention crossed crop boundary");
#ifdef LUNAFLUX_TEST_IMAGE_SCATTER
  counts[0]=1; counts[3]=3;
  int32_t causal[3]={35,0,68},feature_count=2;
  std::vector<int32_t> map(4096,-1);map[35]=1;map[68]=0;
  std::vector<uint16_t> features(2*2048),residual(3*2048,0xdeadU);
  for(int i=0;i<int(features.size());++i)features[i]=uint16_t(i*31+0x7f81U);
  const auto scatter=[&]{for(int row=0;row<3;++row)for(int lane=0;lane<128;++lane){
    blockIdx={row,0,0};threadIdx={lane,0,0};blockDim={128,1,1};
    lunaflux_minicpm_v47_image_scatter(counts,causal,map.data(),&feature_count,features.data(),residual.data());
  }};
  scatter();
  for(int c=0;c<2048;++c){
    require(residual[c]==features[2048+c],"prefill feature map at causal position35");
    require(residual[2048+c]==0xdeadU,"text row was overwritten");
    require(residual[4096+c]==features[c],"prefill feature map at causal position68");
  }
  counts[0]=0;std::fill(residual.begin(),residual.end(),uint16_t(0xbeefU));scatter();
  for(auto word:residual)require(word==0xbeefU,"decode image insertion");
  counts[0]=1;feature_count=0;scatter();
  for(auto word:residual)require(word==0xbeefU,"cleared feature header did not disable insertion");
  feature_count=2;map[68]=2;scatter();
  for(int c=0;c<2048;++c)require(residual[4096+c]==0xbeefU,"feature bound guard");
  std::puts("Generated scatter host probe passed: causal chunk indices, raw BF16 words, text/decode preservation and cleared feature count.");
#endif
  std::puts("Generated vision source host probe passed: patch/position/window/mean and head72 BF16 attention. No CUDA execution.");
}
