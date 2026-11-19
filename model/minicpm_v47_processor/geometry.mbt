///|
fn nearest_even(value : Double) -> Int32 {
  let lower = value.floor().to_int()
  let rest = value - lower.to_double()
  if rest > 0.5 || (rest == 0.5 && lower % 2 != 0) {
    lower + 1
  } else {
    lower
  }
}

///|
fn ensure_divide(value : Double, divisor : Int32) -> Int32 raise {
  let rounded = nearest_even(value / divisor.to_double()).max(1).to_int64() *
    divisor.to_int64()
  guard rounded <= 2147483647L else { fail("image dimension exceeds Int32") }
  rounded.to_int()
}

///|
fn resized(height : Double, width : Double, upscale : Bool) -> Size raise {
  let (h, w) = if height * width > 448.0 * 448.0 || upscale {
    let aspect = width / height
    let h = (448.0 / aspect.sqrt()).to_int()
    (h.to_double(), (h.to_double() * aspect).to_int().to_double())
  } else {
    (height, width)
  }
  // Both 4x and 16x profiles resize to multiples of four encoder patches.
  { height: ensure_divide(h, 56), width: ensure_divide(w, 56), }
}

///|
fn sliced_grid(
  height : Int32,
  width : Int32,
  max_slices : Int32,
) -> (Int32, Int32) {
  let ratio = height.to_double() * width.to_double() / (448.0 * 448.0)
  let multiple = ratio.ceil().to_int().min(max_slices)
  if multiple <= 1 {
    return (0, 0)
  }
  let log_ratio = @math.ln(width.to_double() / height.to_double())
  let mut best = (1, 1)
  let mut minimum = 1.0e300
  for count in (multiple - 1)..<=(multiple + 1) {
    if count == 1 || count > max_slices {
      continue
    }
    for rows in 1..<=count {
      if count % rows != 0 {
        continue
      }
      let columns = count / rows
      let error = (log_ratio - @math.ln(columns.to_double() / rows.to_double())).abs()
      if error < minimum || (error == minimum && rows > best.0) {
        best = (rows, columns)
        minimum = error
      }
    }
  }
  best
}

///|
/// Default selected processor: patch14, resolution448, slicing enabled, at
/// most nine slices. Reducing max_slices to one explicitly disables detail.
pub fn Plan::build(
  spec : @model.Spec,
  height~ : Int32,
  width~ : Int32,
  max_slices~ : Int32 = 9,
) -> Self raise {
  guard height > 0 &&
    width > 0 &&
    height.to_int64() * width.to_int64() <= 2147483647L / 3L &&
    max_slices >= 1 &&
    max_slices <= 9 else {
    fail("invalid RGB image geometry or slice limit")
  }
  let grid = sliced_grid(height, width, max_slices)
  let source = resized(height.to_double(), width.to_double(), grid.0 == 0)
  let crops = [{ canvas: source, top: 0, left: 0, size: source, }]
  if grid.0 > 0 {
    let refine_h = ensure_divide(height.to_double(), grid.0)
    let refine_w = ensure_divide(width.to_double(), grid.1)
    let patch = resized(
      refine_h.to_double() / grid.0.to_double(),
      refine_w.to_double() / grid.1.to_double(),
      true,
    )
    guard patch.height.to_int64() * grid.0.to_int64() <= 2147483647L &&
      patch.width.to_int64() * grid.1.to_int64() <= 2147483647L else {
      fail("detail canvas dimension exceeds Int32")
    }
    let canvas = { height: patch.height * grid.0, width: patch.width * grid.1, }
    for row in 0..<grid.0 {
      for column in 0..<grid.1 {
        crops.push({
          canvas,
          top: row * patch.height,
          left: column * patch.width,
          size: patch,
        })
      }
    }
  }
  let merge = match spec.downsample() {
    FourX => 2
    SixteenX => 4
  }
  let mut columns = 0L
  let mut token_count = 0L
  for crop in crops {
    let pixels = crop.size.height.to_int64() * crop.size.width.to_int64()
    columns += pixels / 14L
    token_count += pixels / (14 * merge * 14 * merge).to_int64()
  }
  guard columns * 84L <= 2147483647L &&
    token_count <= spec.decoder().context_length().to_int64() else {
    fail("image preparation exceeds model or host capacity")
  }
  {
    input: { height, width, },
    grid,
    crops: ReadOnlyArray::from_array(crops),
    markers: spec.markers(),
    merge,
    columns: columns.to_int(),
    token_count: token_count.to_int(),
  }
}
