///|
fn round_bf16(value : Float) -> Float {
  let bits = value.reinterpret_as_uint()
  Float::reinterpret_from_uint(
    (bits + 0x7fffU + ((bits >> 16) & 1U)) & 0xffff0000U,
  )
}

///|
/// Numerical oracle for the pinned Transformers4.57 learned-position contract:
/// coefficients, products and each left-associated add round independently.
pub fn interpolate_bf16(
  corners : ArrayView[Float],
  dy : Float,
  dx : Float,
) -> Float raise VisionError {
  guard corners.length() == 4 &&
    dy >= 0.0F &&
    dy <= 1.0F &&
    dx >= 0.0F &&
    dx <= 1.0F else {
    raise InvalidGeometry
  }
  let coefficients = [
    (1.0F - dy) * (1.0F - dx),
    (1.0F - dy) * dx,
    dy * (1.0F - dx),
    dy * dx,
  ]
  let terms = Array::makei(4, i => {
    round_bf16(round_bf16(corners[i]) * round_bf16(coefficients[i]))
  })
  round_bf16(round_bf16(round_bf16(terms[0] + terms[1]) + terms[2]) + terms[3])
}

///|
/// Bilinear indices follow48×48 learned table endpoints, independently of
/// merge-group row order. Temporal frames reuse the same spatial coordinates.
pub fn VisionPlan::position_stencil(
  self : Self,
  row : Int,
) -> ReadOnlyArray[(Int, Float)] raise VisionError {
  let (_, _, y, x) = self.position(row)
  let (grid_index, _, _, _) = self.merged_position(row / 4)
  let grid = self.grids[grid_index]
  let fy = Float::from_int(y) * 47.0F / Float::from_int(grid.height - 1)
  let fx = Float::from_int(x) * 47.0F / Float::from_int(grid.width - 1)
  let y0 = fy.to_int()
  let x0 = fx.to_int()
  let y1 = Int::min(y0 + 1, 47)
  let x1 = Int::min(x0 + 1, 47)
  let dy = fy - Float::from_int(y0)
  let dx = fx - Float::from_int(x0)
  [
    (y0 * 48 + x0, (1.0F - dy) * (1.0F - dx)),
    (y0 * 48 + x1, (1.0F - dy) * dx),
    (y1 * 48 + x0, dy * (1.0F - dx)),
    (y1 * 48 + x1, dy * dx),
  ]
}
