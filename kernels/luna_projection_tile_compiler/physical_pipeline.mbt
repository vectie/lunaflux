///|
/// Fully bound producer/storage/consumer program. Construction happens before
/// terminal emission; no consumer may independently reconstruct the ring.
pub struct ProjectionPhysicalPipeline {
  schedule : ProjectionFoldPipeline
  operands : @physical_tile_ir.OperandStorage
  fragments : @physical_tile_ir.FragmentProgram
  effects : ProjectionFoldEffectPlan
  distribution : ProjectionProductFoldDistribution
  sibling_reuse : Bool
  owners_per_group : Int
  transfer_segments : ReadOnlyArray[ProjectionTransferSegment]
} derive(Debug, Eq)

///|
pub extend ProjectionPhysicalPipeline with @debug.Debug::{to_repr}

///|
pub extend ProjectionPhysicalPipeline with Eq::{equal, not_equal}

///|
pub fn ProjectionPhysicalPipeline::new(
  schedule : ProjectionFoldPipeline,
  layout : @physical_tile_ir.OperandLayout,
  consumer_groups~ : Int,
  owners_per_group~ : Int,
  products~ : Int,
  execution : ProjectionTransferExecution,
  columns? : Int,
  output_tiles? : Int,
) -> ProjectionPhysicalPipeline? {
  guard products == 1 || products == 2 else { return None }
  let output_tiles = output_tiles.unwrap_or(schedule.output_tiles())
  guard output_tiles > 0 && output_tiles <= schedule.output_tiles() else {
    return None
  }
  let reuse = if products == 2 {
    partition_projection_sibling_reuse(
      row_tiles=schedule.row_width() / schedule.tile_width(),
      output_tiles~,
      consumer_groups~,
    )
  } else {
    None
  }
  guard (if columns is Some(_) {
      partition_projection_column_fold(
        schedule.row_width() / schedule.tile_width(),
        output_tiles,
        consumer_groups,
      )
    } else {
      partition_projection_product_fold(
        row_tiles=schedule.row_width() / schedule.tile_width(),
        output_tiles~,
        consumer_groups~,
      )
    })
    is Some(ordinary) else {
    return None
  }
  let distribution = reuse.unwrap_or(ordinary)
  let weight_rows = output_tiles * schedule.tile_width()
  guard owners_per_group > 0 && consumer_groups <= 2147483647 / owners_per_group else {
    return None
  }
  guard @physical_tile_ir.OperandStorage::new(
      layout,
      if products == 2 {
        [schedule.row_width(), weight_rows, weight_rows]
      } else {
        [schedule.row_width(), weight_rows]
      },
      owners=consumer_groups * owners_per_group,
      stages=schedule.stage_count(),
    )
    is Some(operands) else {
    return None
  }
  guard layout.geometry().0 == schedule.transfer_width() else { return None }
  let fragment_rows = distribution.rows_per_group()
  guard fragment_rows > 0 &&
    fragment_rows <= schedule.row_width() / schedule.tile_width() else {
    return None
  }
  let fragment_columns = columns.unwrap_or(
    if reuse is Some(_) {
      1
    } else {
      products
    },
  )
  guard columns is None ||
    (fragment_columns > 0 && fragment_columns <= schedule.output_tiles()) else {
    return None
  }
  guard schedule.fragment_program(rows=fragment_rows, columns=fragment_columns)
    is Some(fragments) else {
    return None
  }
  let transfer_segments = if products == 2 {
    guard partition_projection_transfer_segments(
        [
          operands.region(0).unwrap().ownership().geometry().0,
          operands.region(1).unwrap().ownership().geometry().0,
          operands.region(2).unwrap().ownership().geometry().0,
        ],
        workers=consumer_groups * owners_per_group,
      )
      is Some(segments) else {
      return None
    }
    segments
  } else {
    []
  }
  Some({
    schedule,
    operands,
    fragments,
    distribution,
    effects: schedule.effect_plan(
      if reuse is Some(_) {
        SerialTransfer
      } else {
        execution
      },
    ),
    sibling_reuse: reuse is Some(_),
    owners_per_group,
    transfer_segments: ReadOnlyArray::from_array(transfer_segments),
  })
}
