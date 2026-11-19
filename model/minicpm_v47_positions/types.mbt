///|
pub(all) enum Modality {
  Text
  Image
  Video
} derive(Eq, Debug)

///|
pub extend Modality with Eq::{equal, not_equal}

///|
pub extend Modality with @debug.Debug::{to_repr}

///|
/// Patch grid before the selected 2x2 or 4x4 spatial merge.
pub(all) struct Grid {
  height : Int32
  width : Int32
}

///|
/// One unpadded prompt. Row-major T/H/W coordinates are immutable; linear cache
/// positions are supplied separately when staging a frame. Construction runs
/// once during request preparation, not in the token-step path.
pub struct Sequence {
  priv coordinates : ReadOnlyArray[Int32]
  priv delta : Int32
  priv capacity : Int32
}

///|
pub fn Sequence::prompt_tokens(self : Self) -> Int32 {
  self.coordinates.length() / 3
}

///|
pub fn Sequence::coordinates(self : Self) -> ReadOnlyArray[Int32] {
  self.coordinates
}

///|
pub fn Sequence::rope_delta(self : Self) -> Int32 {
  self.delta
}

///|
/// Stage a prompt chunk into caller-owned host buffers. No allocation and no
/// padding; callers compact batched prompts before constructing a Sequence.
pub fn Sequence::write_prompt_into(
  self : Self,
  start~ : Int32,
  rows~ : Int32,
  canvas : FixedArray[Int32],
  causal : FixedArray[Int32],
) -> Unit raise {
  guard start >= 0 &&
    rows > 0 &&
    start.to_int64() + rows.to_int64() <= self.prompt_tokens().to_int64() &&
    canvas.length().to_int64() >= rows.to_int64() * 3L &&
    causal.length() >= rows else {
    fail("invalid prompt position buffers")
  }
  for row in 0..<rows {
    causal[row] = start + row
    for axis in 0..<3 {
      canvas[row * 3 + axis] = self.coordinates[(start + row) * 3 + axis]
    }
  }
}

///|
/// Stage appended text. Rotary coordinates advance from the prompt's maximum
/// canvas coordinate, while causal addresses keep the actual token count.
pub fn Sequence::write_decode_into(
  self : Self,
  causal_start~ : Int32,
  rows~ : Int32,
  canvas : FixedArray[Int32],
  causal : FixedArray[Int32],
) -> Unit raise {
  let last = causal_start.to_int64() + rows.to_int64() - 1L
  let first_rotary = causal_start.to_int64() + self.delta.to_int64()
  guard causal_start >= self.prompt_tokens() &&
    rows > 0 &&
    last <= 2147483647L &&
    first_rotary >= 0L &&
    last + self.delta.to_int64() < self.capacity.to_int64() &&
    canvas.length().to_int64() >= rows.to_int64() * 3L &&
    causal.length() >= rows else {
    fail("invalid decode position buffers")
  }
  for row in 0..<rows {
    causal[row] = causal_start + row
    let position = (first_rotary + row.to_int64()).to_int()
    for axis in 0..<3 {
      canvas[row * 3 + axis] = position
    }
  }
}

///|
priv struct Span {
  start : Int32
  end : Int32
}

///|
priv struct VisualFrame {
  modality : Modality
  thumbnail : Span
  slices : Array[Span]
}

///|
priv struct VisualGroup {
  start : Int32
  frames : Array[VisualFrame]
}
