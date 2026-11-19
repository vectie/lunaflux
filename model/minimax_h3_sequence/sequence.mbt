///|
/// Post-encoder conditioning blocks in request order. Counts are token rows
/// after projection, not raw media dimensions. A video-bearing reference places
/// its audio rows before its visual rows, matching the H3 packed contract.
pub(all) enum ConditionBlock {
  Image(Int32)
  Audio(Int32)
  Video(audio_rows~ : Int32, video_rows~ : Int32)
}

///|
pub(all) enum SegmentKind {
  Text
  ConditionVideo
  ConditionAudio
  TargetAudio
  TargetVideo
} derive(Eq)

///|
pub struct Segment {
  kind : SegmentKind
  start : Int32
  count : Int32
} derive(Eq)

///|
pub(all) suberror SequenceError {
  InvalidShape
  InvalidConditioning
  RowLimitExceeded
}

///|
pub struct Sequence {
  priv assembly : @packed_rows_plan.Plan
  priv segments : ReadOnlyArray[Segment]
  priv used : Int32
}

///|
/// One CFG branch: text, ordered conditioning blocks, target audio, target
/// video, then padding to 64 rows. This is the structural layout only: visual
/// anchors, rotary coordinates, timestep values and attention masks remain
/// separate numerical metadata and must use these same row ranges.
pub fn Sequence::new(
  hidden_width~ : Int32,
  text_rows~ : Int32,
  video_rows~ : Int32,
  audio_rows~ : Int32,
  conditioning~ : ArrayView[ConditionBlock],
  max_rows~ : Int32,
) -> Self raise SequenceError {
  guard hidden_width > 0 && text_rows > 0 && video_rows > 0 && audio_rows > 0 else {
    raise InvalidShape
  }
  guard max_rows > 0 && max_rows <= 268435455 else { raise RowLimitExceeded }
  let segments : Array[Segment] = []
  fn append(
    kind : SegmentKind,
    count : Int32,
    cursor : Int32,
  ) -> Int32 raise SequenceError {
    guard count >= 0 else { raise InvalidConditioning }
    guard count <= max_rows - cursor else { raise RowLimitExceeded }
    if count > 0 {
      segments.push({ kind, start: cursor, count, })
    }
    cursor + count
  }
  let initial = append(Text, text_rows, 0)
  let prefix = for block in conditioning; cursor = initial {
    let next = match block {
      Image(rows) => {
        guard rows > 0 else { raise InvalidConditioning }
        append(ConditionVideo, rows, cursor)
      }
      Audio(rows) => append(ConditionAudio, rows, cursor)
      Video(audio_rows~, video_rows~) => {
        guard video_rows > 0 else { raise InvalidConditioning }
        let next = append(ConditionAudio, audio_rows, cursor)
        append(ConditionVideo, video_rows, next)
      }
    }
    continue next
  } nobreak {
    cursor
  }
  let audio_end = append(TargetAudio, audio_rows, prefix)
  let used = append(TargetVideo, video_rows, audio_end)
  let padding = (64 - used % 64) % 64
  guard padding <= max_rows - used else { raise RowLimitExceeded }
  let total_rows = used + padding
  let assembly = @packed_rows_plan.Plan::new(
    width=hidden_width,
    total_rows~,
    conditioning=Array::makei(prefix, i => i),
    audio=Array::makei(audio_rows, i => prefix + i),
    video=Array::makei(video_rows, i => audio_end + i),
    max_elements=total_rows.to_int64() * hidden_width.to_int64(),
  ) catch {
    _ => raise InvalidShape
  }
  { assembly, segments: ReadOnlyArray::from_array(segments), used, }
}

///|
pub fn Sequence::assembly(self : Self) -> @packed_rows_plan.Plan {
  self.assembly
}

///|
pub fn Sequence::segments(self : Self) -> ReadOnlyArray[Segment] {
  self.segments
}

///|
/// Attention sequence boundaries distinguish live rows and padded rows.
pub fn Sequence::used_rows(self : Self) -> Int32 {
  self.used
}
