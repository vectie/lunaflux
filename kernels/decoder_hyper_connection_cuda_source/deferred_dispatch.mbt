///|
pub fn deferred_post_pre_dispatch_source(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
  dispatch : @physical.DeferredPostPreDispatch,
  symbol~ : String,
) -> Bytes raise {
  let text = StringBuilder()
  let arms = StringBuilder()
  for index, (last, group) in dispatch.buckets() {
    let name = "\{symbol}_bucket\{index}"
    let source = @utf8.decode(
      deferred_post_pre_source(
        producer,
        consumer,
        controls_per_group=group,
        symbol=name,
      ),
    )
    text.write_string(
      source.replace_all(
        old="extern \"C\" __global__ void",
        new="__device__ __forceinline__ void",
      ),
    )
    text <+ "\n"
    arms <+
      "if(counts[3]<=\{last}) { \{name}(counts,post,combine,residual,branch,output,weights,scratch); return; }\n"
  }
  text <+
    "extern \"C\" __global__ void \{symbol}(const int32_t *counts,const float *post,const float *combine,const __nv_bfloat16 *residual,const __nv_bfloat16 *branch,__nv_bfloat16 *output,const __nv_bfloat16 *weights,float *scratch) {\n\{arms.to_string()}}\n"
  @utf8.encode(text.to_string())
}
