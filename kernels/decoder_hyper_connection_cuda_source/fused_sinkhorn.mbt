///|
fn fused_sinkhorn_body(
  precision : @precision.HyperConnectionPrecision,
  prefix : String,
) -> String {
  let (width, _, _) = precision.geometry()
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.control_epsilon(),
  )
  let column_sum = match precision.matrix_sum_law() {
    OrderedFloat32 =>
      "float sum=0.0f; for(int destination=0;destination<\{width};++destination) sum=__fadd_rn(sum,local_combination[destination*\{width}+source]);"
    ButterflyFloat32 =>
      "const float sum=\{prefix}_matrix_sum(local_combination+source,\{width});"
  }
  let row_sum = match precision.matrix_sum_law() {
    OrderedFloat32 =>
      "float sum=0.0f; for(int source=0;source<\{width};++source) sum=__fadd_rn(sum,local_combination[destination*\{width}+source]);"
    ButterflyFloat32 =>
      "const float sum=\{prefix}_matrix_sum(local_combination+destination*\{width},1);"
  }
  let normalize = control_expression(SinkhornNormalize, [
    "local_combination[index]", "sum", eps,
  ])
  (
    $|      bool positive=true;
    $|      for(int i=0;i<\{width*width};++i) {float value=initial_combination[matrix_base+i]; local_combination[i]=value; if(!isfinite(value)||value<=0.0f) positive=false;}
    $|      if(positive) {
    $|        for(int source=0;source<\{width};++source) {\{column_sum} for(int destination=0;destination<\{width};++destination) {int index=destination*\{width}+source;local_combination[index]=\{normalize};}}
    $|        for(int iteration=1;iteration<\{precision.sinkhorn_iterations()};++iteration) {
    $|          for(int destination=0;destination<\{width};++destination) {\{row_sum} for(int source=0;source<\{width};++source) {int index=destination*\{width}+source;local_combination[index]=\{normalize};}}
    $|          for(int source=0;source<\{width};++source) {\{column_sum} for(int destination=0;destination<\{width};++destination) {int index=destination*\{width}+source;local_combination[index]=\{normalize};}}
    $|        }
    $|        for(int i=0;i<\{width*width};++i) combination[matrix_base+i]=local_combination[i];
    $|      }
  )
}

///|
pub fn fused_control_tail_source(
  precision : @precision.HyperConnectionPrecision,
  symbol_prefix~ : String,
) -> Bytes raise {
  let fusion = @physical.ControlTailFusion::new(precision)
  let (_, workers, _) = fusion.geometry()
  let schedule = @physical.ParallelControlPlan::new(precision)
  control_tiles_source(
    precision,
    workers,
    workers,
    true,
    schedule.fuses_normalization(),
    symbol_prefix~,
    fused_sinkhorn=true,
  )
}
