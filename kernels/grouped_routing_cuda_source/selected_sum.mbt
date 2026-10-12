///|
/// Neutral padding preserves all logical slots; descending halves specify
/// arithmetic only. Consumers select device execution geometry independently.
pub fn selected_weight_butterfly_source(
  width~ : Int,
  value_expression~ : String,
  output_symbol~ : String,
) -> String raise GroupedRoutingCudaSourceError {
  guard width > 0 && width <= 256 else { raise InvalidExpertGeometry }
  let extent = for size = 1; size < width; {
    continue size * 2
  } nobreak {
    size
  }
  "  { float selected_partial[\{extent}]; for(int slot=0;slot<\{extent};++slot) selected_partial[slot]=slot<\{width} ? \{value_expression} : 0.0f; for(int span=\{extent / 2};span>0;span>>=1) for(int slot=0;slot<span;++slot) selected_partial[slot]=__fadd_rn(selected_partial[slot],selected_partial[slot+span]); \{output_symbol}=selected_partial[0]; }\n"
}
