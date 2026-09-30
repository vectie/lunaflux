///|
/// Default CUDA transport preference, applied once before effect retention.
fn projection_operand_copy_is_async(width : Int) -> Bool {
  width >= 64
}

///|
/// CUDA supplies only instruction geometry and transport capability. All
/// producer, fold and lifetime relations are retained by the generic pass.
fn projection_refine_physical(
  compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  sibling_ownership? : @luna_projection_tile_compiler.ProjectionSiblingOwnership = SplitSiblingPlanes,
  sibling_transport? : @luna_projection_tile_compiler.ProjectionSiblingTransport = BaselineSiblingTransport,
) -> @luna_projection_tile_compiler.PhysicalProjectionTileCompilation raise ProjectionAotError {
  let physical = compilation.refine_physical(
    width => {
      ({ columns: width, } : CudaProjectionTransportLayout).checked_plan()
    },
    width => {
      if projection_operand_copy_is_async(width) {
        OverlappedTransfer
      } else {
        CompletedPrefetch
      }
    },
    owners_per_group=32,
    sibling_ownership~,
    sibling_transport~,
  )
  guard physical is Some(physical) else { raise Unsupported(Strategy) }
  physical
}
