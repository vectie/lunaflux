///|
/// AOT lowering of an immutable numerical law. The caller supplies a local
/// F32 `product`; all CUDA details remain below the precision IR boundary.
/// Neumaier compensation retains small terms when a later large term cancels
/// the running sum. No widened product or changed BF16 operator boundary.
pub fn dot_accumulator_source(
  law : @precision.DotReductionLaw,
  accumulator~ : String,
) -> (String, String, String) raise @precision.PrecisionPlanError {
  match law {
    StripedTreeFloat32(_) | GroupedVectorFloat32(..) =>
      raise @precision.PrecisionPlanError::NoImplementation
    OrderedFloat32 =>
      (
        "float \{accumulator} = 0.0f;",
        "\{accumulator} = __fadd_rn(\{accumulator}, product);",
        "",
      )
    CompensatedFloat32 =>
      (
        "float \{accumulator} = 0.0f, \{accumulator}_correction = 0.0f;",
        (
          $|const float next = __fadd_rn(\{accumulator}, product);
          $|if (isfinite(next) && isfinite(\{accumulator}) && isfinite(product)) {
          $|  const float lost = fabsf(\{accumulator}) >= fabsf(product)
          $|    ? __fadd_rn(__fsub_rn(\{accumulator}, next), product)
          $|    : __fadd_rn(__fsub_rn(product, next), \{accumulator});
          $|  \{accumulator}_correction = __fadd_rn(\{accumulator}_correction, lost);
          $|} else { \{accumulator}_correction = 0.0f; }
          $|\{accumulator} = next;
        ),
        "\{accumulator} = __fadd_rn(\{accumulator}, \{accumulator}_correction);",
      )
  }
}
