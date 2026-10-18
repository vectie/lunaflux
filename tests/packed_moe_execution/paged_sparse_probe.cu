#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <cstring>
#include "source.cuh"
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
using B=__nv_bfloat16;
template<class T> static T *make(int n){T *p;ck(cudaMalloc(&p,n*sizeof(T)));ck(cudaMemset(p,0,n*sizeof(T)));return p;}
template<class T> static void put(T*p,const std::vector<T>&v){if(!v.empty())ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
template<class T> static std::vector<T> get(T*p,int n){std::vector<T>v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
struct Buffers {
  int *counts=make<int>(5),*hc=make<int>(5),*reset=make<int>(1),*append=make<int>(19),*positions=make<int>(8),*input_pos=make<int>(8);
  int *row_seq=make<int>(8),*offsets=make<int>(5),*pages=make<int>(16),*gens=make<int>(16),*valid=make<int>(32);
  int *raw=make<int>(32),*pool_valid=make<int>(16),*pool_count=make<int>(1),*selected=make<int>(40);
  B *iq=make<B>(64),*hw=make<B>(16),*q=make<B>(64),*ki=make<B>(32),*g=make<B>(32),*k=make<B>(32),*v=make<B>(32);
  B *hi=make<B>(128),*hg=make<B>(128),*hk=make<B>(128),*hv=make<B>(128),*ape=make<B>(8),*pool=make<B>(64),*out=make<B>(64);
  float *scratch=make<float>(256);
  void close(){void *ps[]={counts,hc,reset,append,positions,input_pos,row_seq,offsets,pages,gens,valid,raw,pool_valid,pool_count,selected,iq,hw,q,ki,g,k,v,hi,hg,hk,hv,ape,pool,out,scratch};for(void*p:ps)ck(cudaFree(p));}
};
static B value(int identity,int position,int component,int plane){return __float2bfloat16_rn(float(((identity+1)*29+position*13+component*7+plane*11)%37-18)/16.f);}
struct History {std::vector<B> i,g,k,v; int size()const{return i.size()/4;} void add(int id,int pos){for(int c=0;c<4;c++){i.push_back(value(id,pos,c,0));g.push_back(value(id,pos,c,1));k.push_back(value(id,pos,c,2));v.push_back(value(id,pos,c,3));}}};
static unsigned bits(B x){unsigned short u;memcpy(&u,&x,2);return u;}
int main(){
  Buffers p,d;std::array<History,4> histories;
  const int page_map[4][4]={{5,1,0,7},{3,4,7,0},{6,2,0,7},{5,1,0,7}};
  std::vector<std::vector<std::pair<int,int>>> steps={{{0,3},{1,4}},{{1,2},{0,2},{2,2}},{{0,1},{1,1}},{{2,1},{3,2}},{{3,3},{1,1}}};
  for(int step=0;step<(int)steps.size();step++){
    std::vector<B>iq,hw,q,ki,g,k,v;std::vector<int>pos,row_seq,offsets={0},pages,gens,owners;
    int sequence=0;
    for(auto [id,n]:steps[step]){
      int base=histories[id].size();
      for(int r=0;r<n;r++){
        int at=base+r;owners.push_back(id);pos.push_back(at);row_seq.push_back(sequence);
        for(int c=0;c<8;c++){iq.push_back(value(id,at,c,4));q.push_back(value(id,at,c,5));}
        for(int c=0;c<2;c++)hw.push_back(value(id,at,c,6));
        for(int c=0;c<4;c++){ki.push_back(value(id,at,c,0));g.push_back(value(id,at,c,1));k.push_back(value(id,at,c,2));v.push_back(value(id,at,c,3));}
        histories[id].add(id,at);
      }
      for(int page=0;page<(base+n+3)/4;page++){pages.push_back(page_map[id][page]);gens.push_back(id==3?2:1);}
      offsets.push_back(pages.size());sequence++;
    }
    int rows=pos.size();put(p.counts,std::vector<int>{sequence,0,sequence,rows,0});
    put(p.input_pos,pos);put(p.row_seq,row_seq);put(p.offsets,offsets);put(p.pages,pages);put(p.gens,gens);
    put(p.iq,iq);put(p.hw,hw);put(p.q,q);put(p.ki,ki);put(p.g,g);put(p.k,k);put(p.v,v);
    ph_begin<<<1,256>>>(p.counts,p.reset,p.hc,p.valid,p.append,p.positions,p.input_pos,p.row_seq,p.offsets,p.pages,p.gens);
    ph_copy<<<1,256>>>(p.append,p.ki,p.g,p.k,p.v,p.hi,p.hg,p.hk,p.hv);
    ph_publish<<<1,256>>>(p.append,p.hc,p.valid);
    paged_pool<<<16,128>>>(p.hc,p.hi,p.hg,p.valid,p.ape,p.pool,p.raw,p.pool_valid,p.pool_count);
    paged_index<<<8,128>>>(p.counts,p.positions,p.iq,p.hw,p.pool,p.raw,p.pool_valid,p.pool_count,p.valid,p.selected,p.scratch,p.row_seq,p.offsets,p.pages,p.gens);
    paged_attention_retained_scores_v1<<<16,256>>>(p.counts,p.positions,p.hc+3,p.valid,p.q,p.hk,p.hv,p.selected,p.out,p.scratch,p.row_seq,p.offsets,p.pages,p.gens);
    ck(cudaDeviceSynchronize());if(get(p.append,3)[2]){fprintf(stderr,"append failure\n");return 3;}
    auto actual=get(p.out,rows*8);auto actual_indices=get(p.selected,rows*5);auto physical=get(p.hv,128);auto valid=get(p.valid,32);
    for(int row=0;row<rows;row++)for(int c=0;c<4;c++){
      int id=owners[row],at=pos[row],phys=page_map[id][at/4]*4+at%4;
      if(bits(physical[phys*4+c])!=bits(histories[id].v[at*4+c])||valid[phys]!=(id==3?2:1)){fprintf(stderr,"wrong physical append\n");return 4;}
    }
    int first=0;
    for(auto [id,n]:steps[step]){
      auto &h=histories[id];put(d.counts,std::vector<int>{1,0,1,n,0});put(d.hc,std::vector<int>{0,0,1,h.size(),0});
      std::vector<int> valid_dense(32);for(int i=0;i<h.size();i++)valid_dense[i]=1;put(d.valid,valid_dense);
      put(d.hi,h.i);put(d.hg,h.g);put(d.hk,h.k);put(d.hv,h.v);
      put(d.positions,std::vector<int>(pos.begin()+first,pos.begin()+first+n));
      put(d.iq,std::vector<B>(iq.begin()+first*8,iq.begin()+(first+n)*8));
      put(d.hw,std::vector<B>(hw.begin()+first*2,hw.begin()+(first+n)*2));
      put(d.q,std::vector<B>(q.begin()+first*8,q.begin()+(first+n)*8));
      dense_pool<<<1,1>>>(d.hc,d.hi,d.hg,d.valid,d.ape,d.pool,d.raw,d.pool_valid,d.pool_count);
      dense_index<<<8,128>>>(d.counts,d.positions,d.iq,d.hw,d.pool,d.raw,d.pool_valid,d.pool_count,d.valid,d.selected,d.scratch);
      dense_attention_retained_scores_v1<<<16,256>>>(d.counts,d.positions,d.hc+3,d.valid,d.q,d.hk,d.hv,d.selected,d.out,d.scratch);
      ck(cudaDeviceSynchronize());auto expected=get(d.out,n*8);auto indices=get(d.selected,n*5);
      for(int i=0;i<n*8;i++)if(bits(expected[i])!=bits(actual[first*8+i])){fprintf(stderr,"step=%d request=%d output=%d expected=%u actual=%u\n",step,id,i,bits(expected[i]),bits(actual[first*8+i]));return 5;}
      // Sparse slots may leave neutral holes before an incomplete causal tail.
      // Compare the ordered logical selection, not unused physical padding.
      for(int row=0;row<n;row++){
        std::vector<int> a,b;
        for(int j=0;j<5;j++){if(indices[row*5+j]>=0)a.push_back(indices[row*5+j]);if(actual_indices[(first+row)*5+j]>=0)b.push_back(actual_indices[(first+row)*5+j]);}
        if(a!=b){fprintf(stderr,"selection mismatch step=%d request=%d row=%d\n",step,id,row);return 6;}
      }
      first+=n;
    }
    printf("step=%d rows=%d sequences=%d ordered_logical_indices=exact attention=bitwise physical_append=exact\n",step,rows,sequence);
  }
  p.close();d.close();ck(cudaDeviceReset());puts("released=true");
}
