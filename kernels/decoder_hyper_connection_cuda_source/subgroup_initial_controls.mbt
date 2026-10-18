///|
/// Cell ownership changes, not arithmetic: each softmax maximum and sum uses
/// the same ordered operands (or the declared butterfly) as the serial owner.
fn subgroup_initial_controls(
  precision : @precision.HyperConnectionPrecision,
  prefix : String,
) -> String {
  let (width, _, _) = precision.geometry()
  let controls = width * (width + 2)
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.control_epsilon(),
  )
  let pre_logit = control_vector_affine_source(
    "stream",
    "0",
    precision.control_vector_affine_law(),
  )
  let post_logit = control_vector_affine_source(
    "p",
    "1",
    precision.control_vector_affine_law(),
  )
  let pre = control_expression(PreOffset, ["\{prefix}_sigmoid(pre_logit)", eps])
  let post = control_expression(PostScale, ["\{prefix}_sigmoid(post_logit)"])
  let affine = matrix_affine_source(
    "control_index",
    precision.control_matrix_affine_law(),
  )
  let exponential = control_expression(SoftmaxWeight, [affine, "maximum"])
  let probability = control_expression(SoftmaxPublish, ["cell", "sum", eps])
  let sum = match precision.matrix_sum_law() {
    OrderedFloat32 =>
      "float sum=0.0f; for(int i=0;i<\{width};++i) sum=__fadd_rn(sum,__shfl_sync(0xffffffffu,cell,destination*\{width}+i));"
    ButterflyFloat32 =>
      "float parts[\{width}]; for(int i=0;i<\{width};++i) parts[i]=__shfl_sync(0xffffffffu,cell,destination*\{width}+i); for(int span=\{width/2};span>0;span>>=1) for(int i=0;i<span;++i) parts[i]=__fadd_rn(parts[i],parts[i+span]); float sum=parts[0];"
  }
  (
    $|  if(worker<32) {
    $|    const bool ok=(worker>=\{controls} || isfinite(base[worker])) && (worker>=3 || isfinite(scale[worker]));
    $|    const bool all_valid=__all_sync(0xffffffffu,ok);
    $|    if(worker==0) valid=all_valid;
    $|    if(all_valid) {
    $|      if(worker<\{width}) {
    $|        const int stream=worker,p=\{width}+stream;
    $|        const float pre_logit=\{pre_logit}; pre[stream]=\{pre};
    $|        const float post_logit=\{post_logit}; post[(unsigned long long)row*\{width}+stream]=\{post};
    $|      }
    $|      const bool owns=worker<\{width*width};
    $|      const int destination=owns?worker/\{width}:0, source=owns?worker%\{width}:0;
    $|      float cell=0.0f;
    $|      if(owns) {
    $|        int control_index=2*\{width}+destination*\{width}; float maximum=\{affine};
    $|        for(int i=1;i<\{width};++i) {control_index=2*\{width}+destination*\{width}+i;const float value=\{affine};if(value>maximum) maximum=value;}
    $|        control_index=2*\{width}+destination*\{width}+source; cell=\{exponential};
    $|      }
    $|      \{sum}
    $|      if(owns) initial_combination[(unsigned long long)row*\{width*width}+worker]=\{probability};
    $|    }
    $|  }
  )
}
