// Offline regression probe. Reuse the frozen-recipe loader, not another ABI.
// No timing claim: constant per-row V gives an exact all-element oracle.
#define LUNAFLUX_POLICY_PROBE_LIBRARY
#include "selected_policy_probe.cu"
#undef LUNAFLUX_POLICY_PROBE_LIBRARY

int main(int argc,char** argv) {
  if(argc!=5)return 1; // SPEC WIDE_PREFILL PARTITIONED_PREFILL DECODE
  const auto s=fields(argv[1]);
  const int qh=number(s,"query_heads"),kh=number(s,"key_value_heads");
  const int d=number(s,"head_dimension"),width=number(s,"input_width");
  const int page=number(s,"tokens_per_page"),stride=number(s,"page_stride_values");
  const int max_rows=number(s,"maximum_rows"),max_tokens=number(s,"maximum_tokens");
  const int rows=8,prefill_tokens=106,tokens=prefill_tokens+rows-1;
  if(qh<=0||kh<=0||d<=0||width!=(qh+2*kh)*d||page<=0||max_rows<rows||max_tokens<tokens)return 1;
  CK(cudaSetDevice(0));CK(cudaFree(nullptr));
  Kernel wide(argv[2],max_tokens,max_rows,false,false);
  Kernel tail(argv[3],128,max_rows,false,false,false,false,true);
  Kernel decode(argv[4],8,max_rows,false,true,false,true);
  // The captured tail retains the declared full grid; unlike the ordinary
  // prefill route it is not the narrow 128-token replay envelope.
  tail.gx=tuple(fields(std::string(argv[3])+"/kernel.recipe").at("partial_grid")).at(0);
  std::printf("selected_grids wide=%u,%u,%u tail=%u,%u,%u decode=%u,%u,%u\n",wide.gx,wide.gy,wide.gz,tail.gx,tail.gy,tail.gz,decode.gx,decode.gy,decode.gz);
  const std::vector<int> history{8086,8215,8211,8207,8203,8199,8195,8220};
  std::vector<int> offsets{0},positions,lengths,pages,po{0};
  for(int row=0;row<rows;row++) {
    int n=row==0?prefill_tokens:1,length=history[row]+n;
    lengths.push_back(length);offsets.push_back(offsets.back()+n);
    for(int t=0;t<n;t++)positions.push_back(history[row]+t);
    for(int p=0;p<(length+page-1)/page;p++)pages.push_back(int(pages.size()));
    po.push_back(int(pages.size()));
  }
  if(int(pages.size())>number(s,"maximum_pages"))return 1;
  for(int i=0;i<int(pages.size());i++)pages[i]=(int(pages.size())-1-i+int(pages.size())/3)%int(pages.size());
  std::vector<__nv_bfloat16> x(size_t(tokens)*width,__float2bfloat16_rn(0));
  std::vector<__nv_bfloat16> k(size_t(pages.size())*stride,__float2bfloat16_rn(0)),v=k;
  for(int row=0;row<rows;row++) {
    auto scalar=__float2bfloat16_rn(float(row+1));
    for(int pos=0;pos<lengths[row];pos++)for(int h=0;h<kh;h++)for(int c=0;c<d;c++)
      v[size_t(pages[po[row]+pos/page])*stride+(pos%page*kh+h)*d+c]=scalar;
    for(int t=offsets[row];t<offsets[row+1];t++)for(int h=0;h<kh;h++)for(int c=0;c<d;c++)
      x[size_t(t)*width+(qh+kh+h)*d+c]=scalar;
  }
  auto full=prefill_metadata(offsets,lengths,po,positions,max_rows,max_tokens);
  auto only=prefill_metadata(offsets,lengths,po,positions,max_rows,max_tokens,1);
  Buffer counts(20),pos(positions.size()*4),off(offsets.size()*4),len(lengths.size()*4);
  Buffer page_off(po.size()*4),page_ids(pages.size()*4),meta(full.size()*4);
  Buffer input(x.size()*2),keys(k.size()*2),values_buffer(v.size()*2),out(size_t(tokens)*qh*d*2);
  counts.put(std::vector<int>{1,7,rows,tokens,int(pages.size())});
  pos.put(positions);off.put(offsets);len.put(lengths);page_off.put(po);page_ids.put(pages);
  input.put(x);keys.put(k);values_buffer.put(v);tail.merge_row_offsets=off.p;
  void* matrix_args[]={&counts.p,&pos.p,&meta.p,&len.p,&page_off.p,&page_ids.p,&input.p,&out.p,&keys.p,&values_buffer.p};
  void* decode_args[]={&counts.p,&pos.p,&off.p,&len.p,&page_off.p,&page_ids.p,&input.p,&out.p,&keys.p,&values_buffer.p};
  const char* modes[]={"ordered-tail-with-decode","unified-wide","unified-tail","tail-retain-decode"};
  for(int mode=0;mode<4;mode++) {
    meta.put(mode==0?only:full);
    size_t unwritten=0,bad=0,prefill_bad=0;
    double max_error=0;
    for(int repeat=0;repeat<8;repeat++) {
      float sentinel=repeat%2?1024.0f:-1024.0f;
      out.put(std::vector<__nv_bfloat16>(out.bytes/2,__float2bfloat16_rn(sentinel)));
      if(mode==1)wide.launch(matrix_args);else tail.launch(matrix_args);
      if(mode==0||mode==3)decode.launch(decode_args);
      CK(cudaDeviceSynchronize());
      const auto result=out.read();
      for(int row=0;row<rows;row++)for(int t=offsets[row];t<offsets[row+1];t++)for(int c=0;c<qh*d;c++) {
        float actual=__bfloat162float(result[size_t(t)*qh*d+c]);
        double error=std::abs(double(actual)-double(row+1));
        if(actual==sentinel)unwritten++;
        if(!std::isfinite(actual)||error>0.03125){bad++;if(row==0)prefill_bad++;}
        max_error=std::max(max_error,error);
      }
    }
    std::printf("mode=%s repeats=8 output_elements=%zu unwritten=%zu bad=%zu prefill_bad=%zu maxabs=%.9g\n",modes[mode],out.bytes/2,unwritten,bad,prefill_bad,max_error);
    if(mode==2) {
      if(unwritten!=size_t(7*qh*d*8)||bad!=unwritten||prefill_bad!=0)return 4;
    } else if(bad||unwritten)return 3;
  }
  if(!same(k,keys.read())||!same(v,values_buffer.read()))return 5;
  std::puts("diagnosis=partitioned-merge-omits-decode-rows unchanged_kv=true");
}
