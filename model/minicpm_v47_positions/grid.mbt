///|
fn merged_grid(
  grid : Grid,
  merge : Int32,
  capacity : Int32,
) -> (Int32, Int32) raise {
  guard grid.height > 0 &&
    grid.width > 0 &&
    grid.height % merge == 0 &&
    grid.width % merge == 0 else {
    fail("invalid merged patch grid")
  }
  let h = grid.height / merge
  let w = grid.width / merge
  guard h.to_int64() * w.to_int64() <= capacity.to_int64() else {
    fail("visual grid exceeds position capacity")
  }
  (h, w)
}

///|
// Preserve upstream's FP32 linspace step and symmetric endpoint calculation
// before nearest-even rounding. Exact rational rounding changes some positions
// (15 samples over extent118 put sample7 at59, not58). The offline ATen oracle
// checks CPU results; physical CUDA parity remains a separate qualification.
fn resampled_coordinate(index : Int32, count : Int32, extent : Int32) -> Int32 {
  if count == 1 {
    return 0
  }
  let step = Float::from_int(extent - 1) / Float::from_int(count - 1)
  let distance = if index < count / 2 { index } else { count - index - 1 }
  let product = Float::from_double(step.to_double() * distance.to_double())
  let value = if index < count / 2 {
    product
  } else {
    Float::from_double((extent - 1).to_double() - product.to_double())
  }
  let lower = value.floor().to_int()
  let rest = value.to_double() - lower.to_double()
  if rest > 0.5 || (rest == 0.5 && lower % 2 != 0) {
    lower + 1
  } else {
    lower
  }
}

///|
fn fill_scalar(
  coordinates : Array[Int32],
  start : Int32,
  end : Int32,
  first : Int32,
) -> Unit {
  for token in start..<end {
    let position = first + token - start
    for axis in 0..<3 {
      coordinates[token * 3 + axis] = position
    }
  }
}

///|
fn fill_grid(
  coordinates : Array[Int32],
  span : Span,
  grid : (Int32, Int32),
  origin : Int32,
  canvas : (Int32, Int32),
  offset : (Int32, Int32),
) -> Unit raise {
  let (height, width) = grid
  guard (span.end - span.start).to_int64() ==
    height.to_int64() * width.to_int64() else {
    fail("placeholder count differs from merged crop grid")
  }
  for h in 0..<height {
    for w in 0..<width {
      let token = span.start + h * width + w
      coordinates[token * 3] = origin
      coordinates[token * 3 + 1] = origin +
        offset.0 +
        resampled_coordinate(h, height, canvas.0)
      coordinates[token * 3 + 2] = origin +
        offset.1 +
        resampled_coordinate(w, width, canvas.1)
    }
  }
}
