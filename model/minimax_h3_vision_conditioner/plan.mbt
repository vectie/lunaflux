///|
pub(all) suberror VisionError {
  InvalidGeometry
  BudgetExceeded
} derive(Eq)

///|
/// Processor grid dimensions count temporal/spatial patches, not pixels.
pub(all) struct Grid {
  time : Int
  height : Int
  width : Int
} derive(Eq)

///|
pub struct VisionPlan {
  grids : ReadOnlyArray[Grid]
  rows : Int
  merged_rows : Int
  workspace_bytes : Int64
  output_bytes : Int64
}

///|
pub fn VisionPlan::new(
  grids : ArrayView[Grid],
  max_workspace_bytes~ : Int64,
  max_output_bytes~ : Int64,
) -> Self raise VisionError {
  guard !grids.is_empty() else { raise InvalidGeometry }
  let rows = for grid in grids; total = 0L {
    guard grid.time > 0 &&
      grid.height > 0 &&
      grid.width > 0 &&
      grid.height % 2 == 0 &&
      grid.width % 2 == 0 else {
      raise InvalidGeometry
    }
    let area = grid.height.to_int64() * grid.width.to_int64()
    guard area <= 2147483647L / grid.time.to_int64() else {
      raise InvalidGeometry
    }
    let count = area * grid.time.to_int64()
    guard count <= 2147483647L - total else { raise InvalidGeometry }
    continue total + count
  } nobreak {
    total
  }
  // State, norm, QKV, rotary QKV, attention, branch, block MLP, merger MLP.
  let workspace_bytes = rows * (1152L * 4L + 3456L * 2L + 4304L + 1152L) * 2L
  let output_bytes = rows / 4L * 5120L * 2L * 4L
  guard workspace_bytes <= max_workspace_bytes &&
    output_bytes <= max_output_bytes else {
    raise BudgetExceeded
  }
  {
    grids: ReadOnlyArray::from_array(grids.to_owned()),
    rows: rows.to_int(),
    merged_rows: (rows / 4L).to_int(),
    workspace_bytes,
    output_bytes,
  }
}

///|
pub fn VisionPlan::workspace_regions(
  self : Self,
) -> ReadOnlyArray[(Int64, Int64)] {
  let regions = []
  let mut offset = 0L
  for width in [1152, 1152, 3456, 3456, 1152, 1152, 4304, 1152] {
    let bytes = self.rows.to_int64() * width.to_int64() * 2L
    regions.push((offset, bytes))
    offset += bytes
  }
  ReadOnlyArray::from_array(regions)
}

///|
/// Index0 is final merger; 1/2/3 are deepstack after visual blocks8/16/24.
pub fn VisionPlan::output_offset(
  self : Self,
  index : Int,
) -> Int64 raise VisionError {
  guard index >= 0 && index < 4 else { raise InvalidGeometry }
  index.to_int64() * self.merged_rows.to_int64() * 5120L * 2L
}

///|
/// Patch rows are already merge-group ordered by the processor: [t,h/2,w/2,2,2].
pub fn VisionPlan::position(
  self : Self,
  row : Int,
) -> (Int, Int, Int, Int) raise VisionError {
  guard row >= 0 && row < self.rows else { raise InvalidGeometry }
  for grid in self.grids; start = 0 {
    let area = grid.height * grid.width
    let end = start + grid.time * area
    if row < end {
      let local_row = (row - start) % area
      let group = local_row / 4
      let y = group / (grid.width / 2) * 2 + local_row % 4 / 2
      let x = group % (grid.width / 2) * 2 + local_row % 2
      let frame = start + (row - start) / area * area
      break (frame, frame + area, y, x)
    }
    continue end
  } nobreak {
    raise InvalidGeometry
  }
}

///|
/// Returns grid index, temporal patch index, merged height and width for one
/// visual output row. Text adapter applies its own token-stream position base
/// and per-frame timestamp segmentation to these model-independent coordinates.
pub fn VisionPlan::merged_position(
  self : Self,
  row : Int,
) -> (Int, Int, Int, Int) raise VisionError {
  guard row >= 0 && row < self.merged_rows else { raise InvalidGeometry }
  for index, grid in self.grids; start = 0 {
    let area = grid.height / 2 * (grid.width / 2)
    let end = start + grid.time * area
    if row < end {
      let r = row - start
      break (index, r / area, r % area / (grid.width / 2), r % (grid.width / 2))
    }
    continue end
  } nobreak {
    raise InvalidGeometry
  }
}
