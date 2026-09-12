///|
/// Expose the finite owner's stride to the target compiler. CUDA owns lane
/// identifiers and the unroll directive; the map and its coverage are generic.
fn projection_transfer_map_loop(vectors : Int, lanes : Int) -> String {
  let map = @luna_projection_tile_compiler.ProjectionVectorTransferMap::new(
    vectors~,
    owners=lanes,
  ).unwrap()
  let (extent, stride, _) = map.geometry()
  "#pragma unroll\n        for (int vector = threadIdx.x; vector < \{extent}; vector += \{stride})"
}
