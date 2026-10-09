///|
/// Exact packed vision-row to token-row map. Strict order prevents concurrent
/// scatter writes to one token and preserves the visual encoder's row order.
pub struct VisualPlacement {
  token_positions : ReadOnlyArray[Int]
  rotary_positions : ReadOnlyArray[(Int, Int, Int)]
}

///|
pub fn EncoderConfig::visual_placement(
  self : Self,
  ids : ArrayView[Int],
  token_positions : ArrayView[Int],
  rotary_positions : ArrayView[(Int, Int, Int)],
) -> VisualPlacement raise EncoderConfigError {
  guard ids.length() > 0 &&
    ids.length() <= self.max_positions &&
    ids.length() <= 262144 &&
    rotary_positions.length() == ids.length() &&
    !token_positions.is_empty() else {
    raise InvalidRequest
  }
  let mut visual = 0
  for row, id in ids {
    guard id >= 0 && id < 151936 else { raise InvalidRequest }
    if id == self.image_token || id == self.video_token {
      guard visual < token_positions.length() && token_positions[visual] == row else {
        raise InvalidRequest
      }
      visual += 1
    }
    let (t, h, w) = rotary_positions[row]
    guard t >= 0 &&
      h >= 0 &&
      w >= 0 &&
      t < self.max_positions &&
      h < self.max_positions &&
      w < self.max_positions else {
      raise InvalidRequest
    }
  }
  guard visual == token_positions.length() else { raise InvalidRequest }
  {
    token_positions: ReadOnlyArray::from_array(token_positions),
    rotary_positions: ReadOnlyArray::from_array(rotary_positions),
  }
}
