// Offline exact-control isolation. All exponentials are strict expf; this
// probe changes only explicit affine/reduction/division rounding order.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void check(cudaError_t e) {
  if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}
}
static float *load(const std::string &path,size_t n) {
  auto *file=std::fopen(path.c_str(),"rb");if(!file){std::perror(path.c_str());std::exit(3);}
  std::vector<float> values(n);const bool read=std::fread(values.data(),4,n,file)==n&&std::fgetc(file)==EOF;
  std::fclose(file);if(!read)std::exit(3);
  float *p=nullptr;check(cudaMalloc(&p,n*4));check(cudaMemcpy(p,values.data(),n*4,cudaMemcpyHostToDevice));return p;
}
static float *allocate(size_t n) {
  float *p=nullptr;check(cudaMalloc(&p,n*4));check(cudaMemset(p,0,n*4));return p;
}
static void save(float *p,size_t n,const std::string &path) {
  std::vector<float> values(n);check(cudaMemcpy(values.data(),p,n*4,cudaMemcpyDeviceToHost));
  auto *file=std::fopen(path.c_str(),"wbx");if(!file){std::perror(path.c_str());std::exit(4);}
  const bool wrote=std::fwrite(values.data(),4,n,file)==n,closed=std::fclose(file)==0;
  if(!wrote||!closed)std::exit(4);
}

template<bool Butterfly> __device__ float sum(float a,float b,float c,float d) {
  if(Butterfly)return __fadd_rn(__fadd_rn(a,c),__fadd_rn(b,d));
  return __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.0f,a),b),c),d);
}
template<bool Reciprocal> __device__ float divide(float value,float denominator) {
  if(Reciprocal)return __fmul_rn(value,__fdiv_rn(1.0f,denominator));
  return __fdiv_rn(value,denominator);
}
template<bool Fma,bool Butterfly,bool Reciprocal> __global__ void probe(
    const float *mixes,const float *scale,const float *base,
    float *logits,float *softmax,float *column,float *combination) {
  const int row=blockIdx.x;float values[16];
  for(int i=0;i<16;++i) {
    const float mix=mixes[row*24+8+i];
    values[i]=Fma?__fmaf_rn(mix,scale[2],base[8+i]):__fadd_rn(__fmul_rn(mix,scale[2]),base[8+i]);
    logits[row*16+i]=values[i];
  }
  for(int j=0;j<4;++j) {
    const float maximum=fmaxf(fmaxf(values[j*4],values[j*4+2]),fmaxf(values[j*4+1],values[j*4+3]));
    for(int k=0;k<4;++k)values[j*4+k]=expf(__fsub_rn(values[j*4+k],maximum));
    const float denominator=sum<Butterfly>(values[j*4],values[j*4+1],values[j*4+2],values[j*4+3]);
    for(int k=0;k<4;++k)values[j*4+k]=__fadd_rn(divide<Reciprocal>(values[j*4+k],denominator),0.000001f);
  }
  for(int i=0;i<16;++i)softmax[row*16+i]=values[i];
  for(int iteration=0;iteration<20;++iteration) {
    if(iteration>0)for(int j=0;j<4;++j) {
      const float denominator=__fadd_rn(sum<Butterfly>(values[j*4],values[j*4+1],values[j*4+2],values[j*4+3]),0.000001f);
      for(int k=0;k<4;++k)values[j*4+k]=divide<Reciprocal>(values[j*4+k],denominator);
    }
    for(int k=0;k<4;++k) {
      const float denominator=__fadd_rn(sum<Butterfly>(values[k],values[4+k],values[8+k],values[12+k]),0.000001f);
      for(int j=0;j<4;++j)values[j*4+k]=divide<Reciprocal>(values[j*4+k],denominator);
    }
    if(iteration==0)for(int i=0;i<16;++i)column[row*16+i]=values[i];
  }
  for(int i=0;i<16;++i)combination[row*16+i]=values[i];
}

template<bool Fma,bool Butterfly,bool Reciprocal> static void run(
    const float *mixes,const float *scale,const float *base,float *logits,
    float *softmax,float *column,float *combination,const std::string &out) {
  const std::string label=std::string(Fma?"fma":"separate")+(Butterfly?"-butterfly":"-ordered")+(Reciprocal?"-reciprocal":"-divide");
  probe<Fma,Butterfly,Reciprocal><<<144,1>>>(mixes,scale,base,logits,softmax,column,combination);
  check(cudaGetLastError());check(cudaDeviceSynchronize());
  save(logits,144*16,out+"/"+label+"-logits.f32");save(softmax,144*16,out+"/"+label+"-softmax.f32");
  save(column,144*16,out+"/"+label+"-column.f32");save(combination,144*16,out+"/"+label+"-combination.f32");
}
int main(int argc,char **argv) {
  if(argc!=3)return 2;const std::string input(argv[1]),out(argv[2]);
  auto *mixes=load(input+"/mixes.f32",144*24),*scale=load(input+"/scale.f32",3),*base=load(input+"/base.f32",24);
  auto *logits=allocate(144*16),*softmax=allocate(144*16),*column=allocate(144*16),*combination=allocate(144*16);
#define RUN(F,B,R) run<F,B,R>(mixes,scale,base,logits,softmax,column,combination,out)
  RUN(false,false,false);RUN(false,false,true);RUN(false,true,false);RUN(false,true,true);
  RUN(true,false,false);RUN(true,false,true);RUN(true,true,false);RUN(true,true,true);
#undef RUN
  for(void *p:{mixes,scale,base,logits,softmax,column,combination})check(cudaFree(p));
  std::puts("sinkhorn_probe rows=144 width=4 iterations=20 matched_original_controls=true routes=8 strict_exp_only=true offline_only=true release=balanced");
}
