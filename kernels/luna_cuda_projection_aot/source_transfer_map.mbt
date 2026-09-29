///|
/// Expose the finite owner's stride to the target compiler. CUDA owns lane
/// identifiers and the unroll directive; the map and its coverage are generic.
fn projection_physical_transfer_map_loop(
  ownership : @physical_tile_ir.VectorOwnership,
) -> String {
  let (extent, stride, _) = ownership.geometry()
  "#pragma unroll\n        for (int vector = threadIdx.x; vector < \{extent}; vector += \{stride})"
}
