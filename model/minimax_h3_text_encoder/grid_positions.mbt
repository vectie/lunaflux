///|
pub(all) enum VisualKind {
  Image
  Video
} derive(Eq)

///|
pub struct VisualGrid {
  kind : VisualKind
  time : Int
  height : Int
  width : Int
}

///|
/// Raw patch-grid dimensions, before the vision encoder's spatial merge by 2.
pub fn VisualGrid::new(
  kind~ : VisualKind,
  time~ : Int,
  height~ : Int,
  width~ : Int,
) -> Self raise EncoderConfigError {
  guard time > 0 &&
    height > 0 &&
    width > 0 &&
    height % 2 == 0 &&
    width % 2 == 0 &&
    time.to_int64() * (height / 2).to_int64() <=
    262144L / (width / 2).to_int64() else {
    raise InvalidRequest
  }
  { kind, time, height, width, }
}

///|
/// Qwen3-VL timestamps are ordinary text tokens. A video grid with T temporal
/// patches consumes T separately bracketed frame blocks and resets visual t to
/// zero for each; following text starts one above the maximum THW coordinate.
/// Grids must be in textual media order, matching the packed visual outputs.
pub fn EncoderConfig::placement_from_grids(
  self : Self,
  ids : ArrayView[Int],
  grids : ArrayView[VisualGrid],
) -> VisualPlacement raise EncoderConfigError {
  guard ids.length() > 0 &&
    ids.length() <= self.max_positions &&
    ids.length() <= 262144 &&
    !grids.is_empty() else {
    raise InvalidRequest
  }
  let positions : Array[(Int, Int, Int)] = []
  let visual : Array[Int] = []
  let mut cursor = 0
  let mut base = 0
  for grid in grids {
    let frames = if grid.kind == Video { grid.time } else { 1 }
    let temporal = if grid.kind == Video { 1 } else { grid.time }
    let height = grid.height / 2
    let width = grid.width / 2
    let token = if grid.kind == Video {
      self.video_token
    } else {
      self.image_token
    }
    for _ in 0..<frames {
      while cursor < ids.length() &&
            ids[cursor] != self.image_token &&
            ids[cursor] != self.video_token {
        positions.push((base, base, base))
        base += 1
        cursor += 1
      }
      let count = temporal * height * width
      guard cursor > 0 &&
        cursor < ids.length() &&
        ids[cursor - 1] == self.vision_start_token &&
        count < ids.length() - cursor &&
        ids[cursor + count] == self.vision_end_token else {
        raise InvalidRequest
      }
      for t in 0..<temporal {
        for h in 0..<height {
          for w in 0..<width {
            guard ids[cursor] == token else { raise InvalidRequest }
            positions.push((base + t, base + h, base + w))
            visual.push(cursor)
            cursor += 1
          }
        }
      }
      base += Int::max(temporal, Int::max(height, width))
    }
  }
  while cursor < ids.length() {
    positions.push((base, base, base))
    base += 1
    cursor += 1
  }
  self.visual_placement(ids, visual, positions)
}
