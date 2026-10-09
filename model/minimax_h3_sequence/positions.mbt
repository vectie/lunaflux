///|
pub struct VideoGrid {
  priv frames : Int
  priv height : Int
  priv width : Int
  priv rows : Int
}

///|
pub fn VideoGrid::new(
  frames~ : Int,
  height~ : Int,
  width~ : Int,
  max_rows~ : Int,
) -> Self raise SequenceError {
  guard frames > 0 &&
    height > 0 &&
    width > 0 &&
    height % 2 == 0 &&
    width % 2 == 0 else {
    raise InvalidShape
  }
  guard max_rows > 0 &&
    frames.to_int64() <=
    max_rows.to_int64() / (height / 2).to_int64() / (width / 2).to_int64() else {
    raise RowLimitExceeded
  }
  { frames, height, width, rows: frames * (height / 2) * (width / 2), }
}

///|
pub(all) enum ReferenceGrid {
  ImageGrid(VideoGrid)
  AudioGrid(Int)
  VideoAudioGrid(VideoGrid, Int)
}

///|
pub struct Position3 {
  time : Double
  height : Double
  width : Double
}

///|
pub struct PositionedSequence {
  priv sequence : Sequence
  priv positions : ReadOnlyArray[Position3]
}

///|
fn spatial_axis(dim : Int, other : Int, index : Int) -> Double {
  let area = (dim.to_double() * other.to_double()).sqrt()
  let ratio = dim.to_double() / area
  let left = (1.0 - ratio) / 2.0
  let right = left + ratio
  // Keep linspace's endpoint subtraction and operation order before scaling.
  let step = (right - left) / (dim / 2).to_double()
  (index.to_double() * step + left) * 32.0
}

///|
fn append_video(
  output : Array[Position3],
  grid : VideoGrid,
  origin : Double,
) -> Double {
  for frame in 0..<grid.frames; elapsed = 0.0 {
    for h in 0..<(grid.height / 2) {
      for w in 0..<(grid.width / 2) {
        output.push({
          time: origin + elapsed,
          height: spatial_axis(grid.height, grid.width, h),
          width: spatial_axis(grid.width, grid.height, w),
        })
      }
    }
    let span = frame_span(frame)
    continue elapsed + span
  } nobreak {
    elapsed
  }
}

///|
fn append_audio(
  output : Array[Position3],
  frames : Int,
  grid : VideoGrid,
  origin : Double,
) -> Unit {
  for stream in 0..<2 {
    let width_index = if stream == 0 { 0 } else { grid.width / 2 - 1 }
    let width = spatial_axis(grid.width, grid.height, width_index)
    for t in 0..<frames {
      output.push({ time: origin + t.to_double(), height: 0.0, width, })
    }
  }
}

///|
/// Ref2VA position semantics, including mixed reference blocks. Encoded
/// dimensions are explicit. No FL2VA first/last anchor approximation is used.
/// Audio has two stream-major channels pinned to each spatial grid's extremes.
pub fn PositionedSequence::references(
  target : VideoGrid,
  hidden_width~ : Int,
  text_rows~ : Int,
  audio_frames~ : Int,
  references~ : ArrayView[ReferenceGrid],
  max_rows~ : Int,
) -> Self raise SequenceError {
  guard audio_frames > 0 && audio_frames <= max_rows / 2 else {
    raise InvalidShape
  }
  let blocks : Array[ConditionBlock] = []
  for reference in references {
    match reference {
      ImageGrid(grid) => {
        guard grid.frames == 1 else { raise InvalidConditioning }
        blocks.push(Image(grid.rows))
      }
      AudioGrid(frames) => {
        guard frames >= 0 && frames <= max_rows / 2 else {
          raise InvalidConditioning
        }
        blocks.push(Audio(frames * 2))
      }
      VideoAudioGrid(grid, frames) => {
        guard frames >= 0 && frames <= max_rows / 2 else {
          raise InvalidConditioning
        }
        blocks.push(Video(audio_rows=frames * 2, video_rows=grid.rows))
      }
    }
  }
  let sequence = Sequence::new(
    hidden_width~,
    text_rows~,
    video_rows=target.rows,
    audio_rows=audio_frames * 2,
    conditioning=blocks,
    max_rows~,
  )
  let positions : Array[Position3] = []
  for i in 0..<text_rows {
    positions.push({ time: i.to_double(), height: 0.0, width: 0.0, })
  }
  let origin = for reference in references; origin = text_rows.to_double() {
    let span = match reference {
      ImageGrid(grid) => {
        ignore(append_video(positions, grid, origin))
        1.0
      }
      AudioGrid(frames) => {
        append_audio(positions, frames, target, origin)
        frames.to_double()
      }
      VideoAudioGrid(grid, frames) => {
        append_audio(positions, frames, grid, origin)
        let video_span = append_video(positions, grid, origin)
        if video_span > frames.to_double() {
          video_span
        } else {
          frames.to_double()
        }
      }
    }
    continue origin + span
  } nobreak {
    origin
  }
  append_audio(positions, audio_frames, target, origin)
  ignore(append_video(positions, target, origin))
  while positions.length() < sequence.assembly.rows().length() {
    positions.push({ time: 0.0, height: 0.0, width: 0.0, })
  }
  { sequence, positions: ReadOnlyArray::from_array(positions), }
}

///|
pub fn PositionedSequence::sequence(self : Self) -> Sequence {
  self.sequence
}

///|
pub fn PositionedSequence::positions(self : Self) -> ReadOnlyArray[Position3] {
  self.positions
}
