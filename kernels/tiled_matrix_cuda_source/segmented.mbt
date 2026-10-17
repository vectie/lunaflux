///|
/// A single launch over independent output regions. All CUDA barriers remain
/// uniform within a CTA; virtual tile IDs are local to one matrix. The same
/// dense renderer provides both standalone and fused arithmetic.
pub fn segmented_source(
  plan : @matrix.SegmentedMatrixPlan,
  symbol~ : String,
) -> Bytes raise @precision.PrecisionPlanError {
  let helpers = []
  let arguments = ["const int *counts", "const __nv_bfloat16 *input"]
  let dispatch = []
  for i, segment in plan.segments() {
    let name = symbol + "_region_" + i.to_string()
    helpers.push(
      dense_render(
        segment,
        symbol=name,
        partial_float32=false,
        device_region=true,
      ),
    )
    arguments.push("const __nv_bfloat16 *weight_" + i.to_string())
    arguments.push("__nv_bfloat16 *output_" + i.to_string())
    let start = plan.offsets()[i]
    let count = segment.tile_count()
    dispatch.push(
      "if(blockIdx.x<\{start+count}U) { \{name}(blockIdx.x-\{start},\{count},tile_a,tile_b,result,counts,input,weight_\{i},output_\{i}); return; }",
    )
  }
  let (a, b, c) = plan.scratch_elements()
  let scratch = "__shared__ __align__(32) __nv_bfloat16 tile_a[\{a}],tile_b[\{b}];\n__shared__ __align__(32) float result[\{c}];"
  @utf8.encode(
    helpers.join("\n") +
    "\nextern \"C\" __global__ void \{symbol}(\{arguments.join(",")}) {\n\{scratch}\n\{dispatch.join("\n")}\n}\n",
  )
}
