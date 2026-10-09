///|
pub struct TileAxis {
  starts : ReadOnlyArray[Int]
  extent : Int
  overlaps : ReadOnlyArray[Int]
}

///|
/// Released complete-tile schedule: 256-pixel tiles, minimum 64 overlap,
/// remaining overlap distributed in 16-pixel latent-aligned round-robin units.
pub fn TileAxis::new(pixels : Int) -> Self raise @plan.PlanError {
  guard pixels >= 32 && pixels % 32 == 0 && pixels <= 65536 else {
    raise @plan.PlanError::InvalidGeometry
  }
  if pixels <= 256 {
    return { starts: [0], extent: pixels, overlaps: [], }
  }
  let mut count = (pixels + 255) / 256
  while 256 * count - 64 * (count - 1) < pixels {
    count += 1
  }
  let overlaps = Array::make(count - 1, 64)
  let remaining = (256 * count - 64 * (count - 1) - pixels) / 16
  for i in 0..<remaining {
    overlaps[i % (count - 1)] += 16
  }
  let starts = [0]
  for overlap in overlaps {
    starts.push(starts[starts.length() - 1] + 256 - overlap)
  }
  {
    starts: ReadOnlyArray::from_array(starts),
    extent: 256,
    overlaps: ReadOnlyArray::from_array(overlaps),
  }
}
