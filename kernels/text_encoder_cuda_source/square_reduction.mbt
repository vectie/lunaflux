///|
/// Pure device lowering of the selected numerical law. No model/shape dispatch.
fn square_reduction(law : @precision.SquareReductionLaw) -> (String, String) {
  match law {
    OrderedFloat32 => ("float sum=0.0f;", "sum=__fadd_rn(sum,__fmul_rn(v,v));")
    CompensatedFloat32 =>
      (
        "float sum=0.0f, correction=0.0f;", "const float square=__fmul_rn(v,v); const float adjusted=__fsub_rn(square,correction); const float next=__fadd_rn(sum,adjusted); correction=__fsub_rn(__fsub_rn(next,sum),adjusted); sum=next;",
      )
  }
}
