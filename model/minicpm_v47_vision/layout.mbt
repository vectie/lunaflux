///|
fn spans(boundaries : Array[Int32]) -> Spans {
  let rows = boundaries[boundaries.length() - 1]
  let starts = Array::make(rows, 0)
  let ends = Array::make(rows, 0)
  for segment in 0..<(boundaries.length() - 1) {
    for row in boundaries[segment]..<boundaries[segment + 1] {
      starts[row] = boundaries[segment]
      ends[row] = boundaries[segment + 1]
    }
  }
  {
    starts: ReadOnlyArray::from_array(starts),
    ends: ReadOnlyArray::from_array(ends),
    boundaries: ReadOnlyArray::from_array(boundaries),
  }
}

///|
fn append_windows(
  indices : Array[Int32],
  height : Int32,
  width : Int32,
  base : Int32,
) -> Unit {
  for y in 0..<(height / 2) {
    for x in 0..<(width / 2) {
      for dy in 0..<2 {
        for dx in 0..<2 {
          indices.push(base + (y * 2 + dy) * width + x * 2 + dx)
        }
      }
    }
  }
}

///|
// Explicit F32 bucketize(right=True) profile. Learned positions use 70x70
// nearest buckets, not bilinear interpolation or canvas MRoPE. CPU/CUDA
// upstream arange equivalence at boundary ties is a separate qualification.
fn buckets(size : Int32, side : Int32) -> Array[Int32] {
  let boundaries = Array::makei(side - 1, i => {
    Float::from_double((i + 1).to_double() / side.to_double())
  })
  Array::makei(size, i => {
    let coordinate = Float::from_int(i) / Float::from_int(size)
    let mut index = 0
    while index < boundaries.length() && coordinate >= boundaries[index] {
      index += 1
    }
    index
  })
}

///|
/// Build once from processor target sizes. Inputs can contain several images;
/// each thumbnail/detail crop remains an isolated full-attention segment.
pub fn Layout::build(
  spec : @model.Spec,
  grids : ArrayView[@positions.Grid],
  maximum_patches~ : Int32,
  maximum_crop_patches~ : Int32,
) -> Self raise {
  guard !grids.is_empty() &&
    maximum_patches > 0 &&
    maximum_crop_patches > 0 &&
    maximum_crop_patches <= maximum_patches else {
    fail("invalid vision layout capacities")
  }
  let early = spec.downsample() is SixteenX
  let divisor = if early { 4 } else { 2 }
  let raw_boundaries = [0]
  let encoded_boundaries = [0]
  let output_boundaries = [0]
  let mut total = 0L
  // Validate all dimensions before constructing row-sized metadata.
  for grid in grids {
    let count = grid.height.to_int64() * grid.width.to_int64()
    guard grid.height > 0 &&
      grid.width > 0 &&
      grid.height % divisor == 0 &&
      grid.width % divisor == 0 &&
      count <= maximum_crop_patches.to_int64() &&
      count <= maximum_patches.to_int64() - total else {
      fail("invalid or oversized encoder crop grid")
    }
    total += count
    raw_boundaries.push(total.to_int())
    encoded_boundaries.push(
      if early {
        (total / 4L).to_int()
      } else {
        total.to_int()
      },
    )
    output_boundaries.push((total / (divisor * divisor).to_int64()).to_int())
  }
  let positions = []
  let windows = []
  let merger = []
  let side = spec.vision().position_grid_side
  for index, grid in grids {
    let y = buckets(grid.height, side)
    let x = buckets(grid.width, side)
    for row in 0..<grid.height {
      for column in 0..<grid.width {
        positions.push(y[row] * side + x[column])
      }
    }
    if early {
      append_windows(windows, grid.height, grid.width, raw_boundaries[index])
    }
    append_windows(
      merger,
      grid.height / (if early { 2 } else { 1 }),
      grid.width / (if early { 2 } else { 1 }),
      encoded_boundaries[index],
    )
  }
  let window_boundaries = if early {
    Array::makei(total.to_int() / 4 + 1, i => i * 4)
  } else {
    [0]
  }
  {
    grids: ReadOnlyArray::from_array(grids.to_owned()),
    raw: spans(raw_boundaries),
    encoded: spans(encoded_boundaries),
    windows: spans(window_boundaries),
    position_ids: ReadOnlyArray::from_array(positions),
    window_indices: ReadOnlyArray::from_array(windows),
    merger_indices: ReadOnlyArray::from_array(merger),
    output_boundaries: ReadOnlyArray::from_array(output_boundaries),
    insert_after: if early {
      Some(spec.vision().compression_after_layer)
    } else {
      None
    },
  }
}
