// Offline diagnostic of terminal-row compaction; never linked into serving.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e=(x); if(e!=0) { fprintf(stderr,"CUDA error %d line %d\n",int(e),__LINE__); exit(2); } } while(0)
struct Buffer {
  void *p=nullptr;
  size_t size;
  explicit Buffer(size_t n):size(n) { CK(cudaMalloc(&p,n)); CK(cudaMemset(p,0,n)); }
  ~Buffer() { CK(cudaFree(p)); }
  void upload(const void *source) { CK(cudaMemcpy(p,source,size,cudaMemcpyHostToDevice)); }
  std::vector<unsigned char> bytes() {
    std::vector<unsigned char> result(size);
    CK(cudaMemcpy(result.data(),p,size,cudaMemcpyDeviceToHost));
    return result;
  }
};
void fill(Buffer &buffer,int salt) {
  std::vector<__nv_bfloat16> data(buffer.size/2);
  for(size_t i=0;i<data.size();++i)
    data[i]=__float2bfloat16_rn(float(int((i*17+salt)%31)-15)/128.0f);
  buffer.upload(data.data());
}
struct Graph {
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  Graph(CUfunction function,void **args,int groups,int shared,int repeat,cudaStream_t stream) {
    CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<repeat;++i)
      CK(cuLaunchKernel(function,(151936+groups*16-1)/(groups*16),1,1,
                        groups*32,1,1,shared,stream,args,nullptr));
    CK(cudaStreamEndCapture(stream,&graph));
    CK(cudaGraphInstantiate(&executable,graph,0));
  }
  ~Graph() { CK(cudaGraphExecDestroy(executable)); CK(cudaGraphDestroy(graph)); }
  void run(cudaStream_t stream) {
    CK(cudaGraphLaunch(executable,stream)); CK(cudaStreamSynchronize(stream));
  }
  float time(cudaStream_t stream,int repeat) {
    for(int i=0;i<3;++i) run(stream);
    cudaEvent_t start,end;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(start,stream)); CK(cudaGraphLaunch(executable,stream));
    CK(cudaEventRecord(end,stream)); CK(cudaEventSynchronize(end));
    float ms; CK(cudaEventElapsedTime(&ms,start,end));
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
    return ms*1000/repeat;
  }
};
int main(int argc,char **argv) {
  if(argc!=5) return 1;
  const int groups=atoi(argv[2]),shared=atoi(argv[3]),repeat=atoi(argv[4]);
  if(groups<1||groups>16||shared<1||repeat<1) return 1;
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  cudaStream_t stream; CK(cudaStreamCreate(&stream));
  CUmodule module; CUfunction function;
  CK(cuModuleLoad(&module,argv[1])); CK(cuModuleGetFunction(&function,module,"distribution"));
  {
    Buffer descriptor((10+257)*4),ends(257*4),input(256*1024*2);
    Buffer weight(size_t(151936)*1024*2),output(size_t(256)*151936*2);
    fill(input,3); fill(weight,5);
    void *compact_counts=(char*)descriptor.p+20,*compact_ends=(char*)descriptor.p+40;
    void *original_args[]={&descriptor.p,&ends.p,&input.p,&weight.p,&output.p};
    void *compact_args[]={&compact_counts,&compact_ends,&input.p,&weight.p,&output.p};
    Graph original(function,original_args,groups,shared,repeat,stream);
    Graph compact(function,compact_args,groups,shared,repeat,stream);
    for(int rows:{1,2,8,17,32,64}) {
      std::vector<int> row_ends(257,0);
      for(int row=0;row<rows;++row) row_ends[row+1]=row_ends[row]+1+row%3;
      const int tokens=row_ends[rows]; ends.upload(row_ends.data());
      for(int mode=0;mode<4;++mode) {
        std::vector<int> counts(10+257,0),selected;
        counts[0]=rows; counts[2]=rows; counts[3]=tokens; counts[4]=rows;
        for(int row=0;row<rows;++row)
          if(mode==0||(mode==2&&row==rows-1)||(mode==3&&row%2==1)) {
            selected.push_back(row_ends[row+1]-1);
            counts[10+selected.size()]=row_ends[row+1];
          }
        const int physical=selected.size()==1&&rows>1?2:selected.size();
        counts[5]=physical; counts[7]=physical; counts[8]=tokens; counts[9]=rows;
        descriptor.upload(counts.data());
        CK(cudaMemset(output.p,0xa5,output.size)); original.run(stream);
        auto expected=output.bytes();
        CK(cudaMemset(output.p,0xa5,output.size)); compact.run(stream);
        auto actual=output.bytes();
        for(int token=0;token<256;++token) {
          const bool producing=std::find(selected.begin(),selected.end(),token)!=selected.end();
          for(int byte=0;byte<151936*2;++byte) {
            const size_t index=size_t(token)*151936*2+byte;
            if(actual[index]!=(producing?expected[index]:0xa5)) {
              fprintf(stderr,"mismatch rows=%d mode=%d token=%d byte=%d\n",rows,mode,token,byte);
              return 4;
            }
          }
        }
        float before=original.time(stream,repeat),after=compact.time(stream,repeat);
        printf("rows=%d tokens=%d outputs=%zu physical=%d original_us=%.6f compact_us=%.6f bitwise=true untouched=true\n",
               rows,tokens,selected.size(),physical,before,after);
      }
    }
  }
  CK(cuModuleUnload(module)); CK(cudaStreamDestroy(stream));
  puts("resources_released=true");
}
