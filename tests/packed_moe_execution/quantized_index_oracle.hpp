// Offline independent E4M3/F64 calculation, never linked into serving.
static float e4m3_value(int code) {
  int e=(code>>3)&15,m=code&7;
  return e==0?std::ldexp(float(m),-9):std::ldexp(1.f+float(m)/8.f,e-7);
}
static float quantized_value(float x,float scale) {
  float z=std::min(448.f,std::fabs(x/scale)),best=0.f,error=INFINITY;
  int chosen=0;
  for(int code=0;code<127;++code){
    float v=e4m3_value(code),d=std::fabs(v-z);
    if(d<error||(d==error&&(code&1)==0)){best=v;error=d;chosen=code;}
  }
  (void)chosen;
  return std::copysign(best,x);
}
static void quantized_index_oracle(
    const std::vector<__nv_bfloat16>& original_pools,
    const std::vector<__nv_bfloat16>& queries,
    const std::vector<__nv_bfloat16>& weights,
    const std::vector<float>& scratch,
    const std::vector<int>& selected,int history,int active) {
  std::vector<float> keys((history/4)*128),ks(history/4);
  for(int key=0;key<history/4;++key){
    float mx=0;for(int c=0;c<128;++c)mx=std::max(mx,std::fabs(__bfloat162float(original_pools[key*128+c])));
    ks[key]=std::max(mx,1.e-12f)/448.f;
    for(int c=0;c<128;++c)keys[key*128+c]=quantized_value(__bfloat162float(original_pools[key*128+c]),ks[key]);
  }
  double worst=0;
  for(int row : {0,active-1}) {
    std::vector<float> q(32*128),qs(32);int candidates=(history-active+row+1)/4;
    for(int head=0;head<32;++head){
      float mx=0;for(int c=0;c<128;++c)mx=std::max(mx,std::fabs(__bfloat162float(queries[(row*32+head)*128+c])));
      qs[head]=std::max(mx,1.e-12f)/448.f;
      for(int c=0;c<128;++c)q[head*128+c]=quantized_value(__bfloat162float(queries[(row*32+head)*128+c]),qs[head]);
    }
    std::vector<int> order;
    for(int key=0;key<candidates;++key){
      double score=0,mass=0;
      for(int head=0;head<32;++head){
        double dot=0,absolute=0;
        for(int c=0;c<128;++c){double p=double(q[head*128+c])*keys[key*128+c];dot+=p;absolute+=std::fabs(p);}
        double scale=double(qs[head])*ks[key]/std::sqrt(128.0);
        double weight=__bfloat162float(weights[row*32+head])/std::sqrt(32.0);
        score+=std::max(0.0,dot*scale)*weight;mass+=absolute*scale*std::fabs(weight);
      }
      double actual=scratch[row*4096+key];
      // F32 dot, scale and ordered head reductions: conservative gamma(192)
      // absolute-product bound, not agreement with another framework.
      double bound=mass*(192.0*std::ldexp(1.0,-24))/(1.0-192.0*std::ldexp(1.0,-24))+1.e-8;
      if(!std::isfinite(actual)||std::fabs(actual-score)>bound){
        fprintf(stderr,"quantized score oracle row=%d key=%d actual=%.9g f64=%.9g bound=%.9g\n",row,key,actual,score,bound);exit(12);
      }
      worst=std::max(worst,std::fabs(actual-score));order.push_back(key);
    }
    std::sort(order.begin(),order.end(),[&](int a,int b){float x=scratch[row*4096+a],y=scratch[row*4096+b];return x>y||(x==y&&a<b);});
    for(int rank=0;rank<std::min(candidates,512);++rank)for(int i=0;i<4;++i)
      if(selected[row*2051+rank*4+i]!=order[rank]*4+i){fprintf(stderr,"quantized stable selection failure\n");exit(13);}
  }
  printf("independent_quantized_index_f64=pass maximum_absolute_error=%.9g\n",worst);
}
