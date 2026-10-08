// CPU-only independent double-precision attention oracle for every captured
// cross-backend outlier. It does not change the original acceptance threshold.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
static float f32(uint32_t bits){float f;std::memcpy(&f,&bits,4);return f;}
static double bf16(uint16_t bits){return f32(uint32_t(bits)<<16);}
static void read(FILE* f,void* p,size_t n,size_t size){if(std::fread(p,size,n,f)!=n)std::abort();}
int main(int argc,char** argv){
  if(argc!=2)return 2;
  FILE* f=std::fopen(argv[1],"rb");if(!f)return 3;
  uint64_t n;read(f,&n,1,8);if(!n||n>128)std::abort();
  std::vector<uint64_t> meta(n*8);read(f,meta.data(),meta.size(),8);
  constexpr size_t history_limit=8256,stride=(1+2*history_limit)*128;
  std::vector<uint16_t> data(stride);int reference_pass=0,original_pass=0,reference_closer=0;
  double maximum_reference_error=0,maximum_original_error=0;
  for(size_t r=0;r<n;r++){
    read(f,data.data(),data.size(),2);auto m=meta.data()+r*8;
    if(m[0]!=2||m[4]>history_limit||m[3]>=128)std::abort();
    int history=int(m[4]),dim=int(m[3]);
    std::vector<double> scores(history);double maximum=-INFINITY;
    for(int pos=0;pos<history;pos++){
      double sum=0;for(int d=0;d<128;d++)sum+=bf16(data[d])*bf16(data[128+pos*128+d]);
      scores[pos]=sum/std::sqrt(128.0);maximum=std::max(maximum,scores[pos]);
    }
    double numerator=0,denominator=0;
    for(int pos=0;pos<history;pos++){
      double p=std::exp(scores[pos]-maximum);
      denominator+=p;numerator+=p*bf16(data[128+history_limit*128+pos*128+dim]);
    }
    double truth=numerator/denominator,original=f32(uint32_t(m[5])),reference=f32(uint32_t(m[6]));
    double bound=.01+.02*std::abs(truth),er=std::abs(reference-truth),eo=std::abs(original-truth);
    reference_pass+=std::isfinite(er)&&er<=bound;original_pass+=std::isfinite(eo)&&eo<=bound;
    reference_closer+=er<eo;maximum_reference_error=std::max(maximum_reference_error,er);maximum_original_error=std::max(maximum_original_error,eo);
    std::printf("record=%zu token=%llu head=%llu dim=%llu history=%d truth=%.12g original=%.12g reference=%.12g bound=%.12g original_error=%.12g reference_error=%.12g\n",r,(unsigned long long)m[1],(unsigned long long)m[2],(unsigned long long)m[3],history,truth,original,reference,bound,eo,er);
  }
  if(std::fgetc(f)!=EOF)std::abort();std::fclose(f);
  std::printf("records=%llu reference_pass=%d original_pass=%d reference_closer=%d maximum_reference_error=%.12g maximum_original_error=%.12g\n",(unsigned long long)n,reference_pass,original_pass,reference_closer,maximum_reference_error,maximum_original_error);
}
