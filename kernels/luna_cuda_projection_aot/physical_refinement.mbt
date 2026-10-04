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
  sibling_execution? : @luna_projection_tile_compiler.ProjectionTransferExecution,
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
  // Preserve the measured default at the backend policy boundary, not as a
  // hidden restriction of the generic ownership/transport product.
  match physical.sibling {
    Some(sibling) => {
      let execution = match sibling_execution {
        Some(execution) => Some(execution)
        None if sibling.sibling_transport == SegmentedSiblingTransport =>
          Some(@luna_projection_tile_compiler.SerialTransfer)
        None => None
      }
      match execution {
        Some(execution) => physical.with_sibling_execution(execution)
        None => physical
      }
    }
    None => physical
  }
}
