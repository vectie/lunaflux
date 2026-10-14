///|
/// Reuse the existing shape classifier and pure scalar fold, without producing
/// cache identities or importing external measurements into numerical legality.
pub fn shape_decode_body(
  inner~ : Int,
  columns~ : Int,
  maximum_rows~ : Int,
  left~ : String,
  right~ : String,
  publish~ : String,
  prepare? : String = "",
) -> String {
  let dispatch = try! @strategy.plan_projection_dispatch(
    @strategy.ProjectionShape::new(
      Dense,
      input_width=inner,
      output_width=columns,
    ),
    maximum_rows,
    @strategy.ProjectionCapabilities::new(subgroup_width=32),
  )
  let reduction : @tile_compiler.ProjectionSingleRowReduction = if dispatch
    .decode()
    .kernel_class() ==
    SubgroupGemv {
    StridedPairwiseDot(32)
  } else {
    OrderedDot
  }
  let fold = @tile_compiler.ProjectionScalarFoldPlan::new(
    extent=inner,
    products=1,
    reduction,
  ).unwrap()
  let (_, _, workers) = fold.geometry()
  let combine = fold
    .combination_distances()
    .map(distance => {
      "value=__fadd_rn(value,__shfl_down_sync(0xffffffffU,value,\{distance}));"
    })
    .join("\n")
  let start = if workers == 1 { "0" } else { "lane" }
  (
    $|// Existing projection strategy: decode GEMV; immutable scalar fold.
    $|const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    $|for(int column=blockIdx.x*4+warp;column<\{columns};column+=gridDim.x*4) {
    $|  \{prepare}
    $|  float value=0.0f;
    $|  for(int inner=\{start};inner<\{inner};inner+=\{workers}) {
    $|    value=__fadd_rn(value,__fmul_rn(\{left},\{right}));
    $|  }
    $|  \{combine}
    $|  if(lane==0) { \{publish} }
    $|}
  )
}
