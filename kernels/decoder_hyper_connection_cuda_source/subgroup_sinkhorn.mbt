///|
/// One independent matrix cell per lane. Cross-cell reads use a subgroup,
/// while each cell's sum follows the declared numeric order exactly.
fn subgroup_sinkhorn_body(
  precision : @precision.HyperConnectionPrecision,
) -> String {
  let (width, _, _) = precision.geometry()
  let eps = @luna_cuda_pointwise_aot.canonical_cuda_float_literal(
    precision.control_epsilon(),
  )
  fn sum(axis : String) -> String {
    let index = if axis == "column" {
      "i*\{width}+source"
    } else {
      "destination*\{width}+i"
    }
    match precision.matrix_sum_law() {
      OrderedFloat32 =>
        "float sum=0.0f; for(int i=0;i<\{width};++i) sum=__fadd_rn(sum,__shfl_sync(0xffffffffu,cell,\{index}));"
      ButterflyFloat32 =>
        "float parts[\{width}]; for(int i=0;i<\{width};++i) parts[i]=__shfl_sync(0xffffffffu,cell,\{index}); for(int span=\{width/2};span>0;span>>=1) for(int i=0;i<span;++i) parts[i]=__fadd_rn(parts[i],parts[i+span]); float sum=parts[0];"
    }
  }
  let column_sum = sum("column")
  let row_sum = sum("row")
  let normalize = control_expression(SinkhornNormalize, ["cell", "sum", eps])
  (
    $|  if(worker<32) {
    $|    const bool owns=worker<\{width*width};
    $|    const int destination=owns?worker/\{width}:0, source=owns?worker%\{width}:0;
    $|    const unsigned long long matrix_base=(unsigned long long)row*\{width*width};
    $|    float cell=owns?initial_combination[matrix_base+worker]:1.0f;
    $|    if(__all_sync(0xffffffffu,isfinite(cell)&&cell>0.0f)) {
    $|      {\{column_sum} cell=\{normalize};}
    $|      for(int iteration=1;iteration<\{precision.sinkhorn_iterations()};++iteration) {
    $|        {\{row_sum} cell=\{normalize};}
    $|        {\{column_sum} cell=\{normalize};}
    $|      }
    $|      if(owns) combination[matrix_base+worker]=cell;
    $|    }
    $|  }
  )
}
