///|
pub struct Source {
  source : Bytes
  symbols : ReadOnlyArray[String]
  grids : ReadOnlyArray[(Int, Int)]
}

///|
/// Low-level lowering of immutable F32 media encoder primitives. Every entry
/// has five read operands and one output; unused read operands are ignored.
/// No model names, state lookup, host copies, or dynamic allocation occur here.
pub fn render(graph : @plan.Graph) -> Source raise @plan.PlanError {
  let s = StringBuilder()
  s.write_string(
    "#include <math.h>\n#include <stdint.h>\n#include <cuda_fp16.h>\n",
  )
  let symbols : Array[String] = []
  let grids : Array[(Int, Int)] = []
  for i, node in graph.nodes {
    let symbol = "media_encode_\{i}"
    symbols.push(symbol)
    let (_, count) = node.operation.signature()
    let (grid, threads) = match node.operation {
      WeightNorm(outputs~, ..) => (outputs, 1)
      LayerNorm(batch~, time~, ..) => (batch * time, 1)
      CausalAttention(batch~, heads~, time~, ..) => (batch * heads * time, 1)
      GroupNormSilu(frames~, groups~, ..) => (frames * groups, 1)
      _ => (((count + 255L) / 256L).to_int(), 256)
    }
    grids.push((grid, threads))
    s.write_string(
      "extern \"C\" __global__ void \{symbol}(const float *a, const float *b, const float *c, const float *d, const float *e, float *out) {\n",
    )
    s.write_string(
      "const unsigned long long f=(unsigned long long)blockIdx.x*\{threads}ULL+threadIdx.x;\n",
    )
    s.write_string(body(node.operation))
    s.write_string("\n}\n")
  }
  {
    source: @utf8.encode(s.to_string()),
    symbols: ReadOnlyArray::from_array(symbols),
    grids: ReadOnlyArray::from_array(grids),
  }
}

///|
fn body(op : @plan.Operation) -> String raise @plan.PlanError {
  let (_, count) = op.signature()
  let guard_flat = "if(f>=\{count}ULL)return;\n"
  match op {
    Pad(batch=_, input_time~, output_time~) =>
      guard_flat +
      "const int t=f%\{output_time}ULL; out[f]=t<\{input_time}?a[(f/\{output_time}ULL)*\{input_time}ULL+t]:0.0f;"
    WeightNorm(outputs~, per_output~) =>
      (
        $|if(f>=\{outputs}ULL)return;
        $|float sum=0.0f; const unsigned long long base=f*\{per_output}ULL;
        $|for(int j=0;j<\{per_output};++j)sum=__fadd_rn(sum,__fmul_rn(b[base+j],b[base+j]));
        $|const float scale=__fdiv_rn(a[f],sqrtf(sum));
        $|for(int j=0;j<\{per_output};++j)out[base+j]=__fmul_rn(b[base+j],scale);
      )
    Conv(
      batch=_,
      input_channels~,
      output_channels~,
      input_time~,
      output_time~,
      kernel~,
      dilation~,
      stride~,
      padding~
    ) =>
      guard_flat +
      (
        $|const int t=f%\{output_time}ULL, co=(f/\{output_time}ULL)%\{output_channels}ULL;
        $|const unsigned long long n=f/(\{output_time}ULL*\{output_channels}ULL);
        $|float sum=c[co];
        $|for(int ci=0;ci<\{input_channels};++ci)for(int k=0;k<\{kernel};++k){
        $|const long long ti=(long long)t*\{stride}+k*\{dilation}-\{padding};
        $|if(ti>=0 && ti<\{input_time})sum=__fadd_rn(sum,__fmul_rn(a[(n*\{input_channels}ULL+ci)*\{input_time}ULL+ti],b[((unsigned long long)co*\{input_channels}ULL+ci)*\{kernel}ULL+k]));
        $|}out[f]=sum;
      )
    Snake(batch=_, channels~, time~) =>
      guard_flat +
      (
        $|const float alpha=b[(f/\{time}ULL)%\{channels}ULL];
        $|const float x=a[f], sine=sinf(__fmul_rn(alpha,x));
        $|out[f]=__fadd_rn(x,__fmul_rn(__fdiv_rn(1.0f,__fadd_rn(alpha,1e-9f)),__fmul_rn(sine,sine)));
      )
    Add(..) => guard_flat + "out[f]=__fadd_rn(a[f],b[f]);"
    LayerNorm(batch~, channels~, time~) =>
      (
        $|if(f>=\{batch * time}ULL)return;const int t=f%\{time}ULL;
        $|const unsigned long long base=(f/\{time}ULL)*\{channels}ULL*\{time}ULL+t;
        $|float mean=0.0f;for(int ch=0;ch<\{channels};++ch)mean=__fadd_rn(mean,a[base+(unsigned long long)ch*\{time}ULL]);
        $|mean=__fdiv_rn(mean,\{channels}.0f);float variance=0.0f;
        $|for(int ch=0;ch<\{channels};++ch){float v=__fsub_rn(a[base+(unsigned long long)ch*\{time}ULL],mean);variance=__fadd_rn(variance,__fmul_rn(v,v));}
        $|const float inv=rsqrtf(__fadd_rn(__fdiv_rn(variance,\{channels}.0f),1e-5f));
        $|for(int ch=0;ch<\{channels};++ch){const unsigned long long ix=base+(unsigned long long)ch*\{time}ULL;out[ix]=__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(a[ix],mean),inv),b[ch]),c[ch]);}
      )
    Dense(batch=_, input_channels~, output_channels~, time~) =>
      guard_flat +
      (
        $|const int t=f%\{time}ULL,co=(f/\{time}ULL)%\{output_channels}ULL;
        $|const unsigned long long n=f/(\{time}ULL*\{output_channels}ULL);float sum=c[co];
        $|for(int ci=0;ci<\{input_channels};++ci)sum=__fadd_rn(sum,__fmul_rn(a[(n*\{input_channels}ULL+ci)*\{time}ULL+t],b[(unsigned long long)co*\{input_channels}ULL+ci]));out[f]=sum;
      )
    Qkv(batch=_, channels~, time~) =>
      guard_flat +
      (
        $|const int t=f%\{time}ULL,co=(f/\{time}ULL)%\{channels * 3}ULL;
        $|const unsigned long long n=f/(\{time}ULL*\{channels * 3}ULL);
        $|float sum=co<\{channels}?c[co]:(co<\{channels * 2}?d[co-\{channels}]:e[co-\{channels * 2}]);
        $|for(int ci=0;ci<\{channels};++ci)sum=__fadd_rn(sum,__fmul_rn(a[(n*\{channels}ULL+ci)*\{time}ULL+t],b[(unsigned long long)co*\{channels}ULL+ci]));out[f]=sum;
      )
    CausalAttention(batch~, heads~, head_dim~, time~) =>
      (
        $|if(f>=\{batch * heads * time}ULL)return;
        $|const int q=f%\{time}ULL,h=(f/\{time}ULL)%\{heads}ULL;
        $|const unsigned long long n=f/(\{time}ULL*\{heads}ULL), base=n*\{3 * heads * head_dim}ULL*\{time}ULL;
        $|float acc[\{head_dim}];for(int j=0;j<\{head_dim};++j)acc[j]=0.0f;
        $|float maximum=-INFINITY,denom=0.0f;const float scale=rsqrtf(\{head_dim}.0f);
        $|for(int k=0;k<=q;++k){float score=0.0f;
        $|for(int j=0;j<\{head_dim};++j)score=__fadd_rn(score,__fmul_rn(a[base+(unsigned long long)(h*\{head_dim}+j)*\{time}ULL+q],a[base+(unsigned long long)(\{heads * head_dim}+h*\{head_dim}+j)*\{time}ULL+k]));
        $|score=__fmul_rn(score,scale);float next=fmaxf(maximum,score),old=expf(__fsub_rn(maximum,next)),p=expf(__fsub_rn(score,next));
        $|denom=__fadd_rn(__fmul_rn(denom,old),p);
        $|for(int j=0;j<\{head_dim};++j)acc[j]=__fadd_rn(__fmul_rn(acc[j],old),__fmul_rn(p,a[base+(unsigned long long)(\{2 * heads * head_dim}+h*\{head_dim}+j)*\{time}ULL+k]));maximum=next;}
        $|for(int j=0;j<\{head_dim};++j)out[((n*\{heads}ULL+h)*\{head_dim}ULL+j)*\{time}ULL+q]=__fdiv_rn(acc[j],denom);
      )
    MeanHeadsPool(batch=_, heads~, head_dim~, output_channels~, time~) =>
      guard_flat +
      (
        $|const int t=f%\{time}ULL,cx=(f/\{time}ULL)%\{output_channels}ULL;
        $|const unsigned long long n=f/(\{time}ULL*\{output_channels}ULL);float pooled=0.0f;
        $|for(int j=0;j<\{head_dim / output_channels};++j){float mean=0.0f;
        $|for(int h=0;h<\{heads};++h)mean=__fadd_rn(mean,a[((n*\{heads}ULL+h)*\{head_dim}ULL+cx*\{head_dim / output_channels}+j)*\{time}ULL+t]);
        $|pooled=__fadd_rn(pooled,__fdiv_rn(mean,\{heads}.0f));}out[f]=__fdiv_rn(pooled,\{head_dim / output_channels}.0f);
      )
    GeluMultiply(..) =>
      guard_flat +
      (
        $|const float x=a[f],cube=__fmul_rn(__fmul_rn(x,x),x);
        $|const float inner=__fmul_rn(0.7978845608028654f,__fadd_rn(x,__fmul_rn(0.044715f,cube)));
        $|out[f]=__fmul_rn(__fmul_rn(__fmul_rn(0.5f,x),__fadd_rn(1.0f,tanhf(inner))),b[f]);
      )
    NormalizeRows(batch=_, channels~, time~) =>
      guard_flat +
      (
        $|const int ch=f%\{channels}ULL,t=(f/\{channels}ULL)%\{time}ULL;
        $|const unsigned long long n=f/(\{channels}ULL*\{time}ULL);
        $|out[f]=__fdiv_rn(__fsub_rn(a[(n*\{channels}ULL+ch)*\{time}ULL+t],b[ch]),c[ch]);
      )
    _ => video_body(op)
  }
}
