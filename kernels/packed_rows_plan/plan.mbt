///|
pub(all) enum Source {
  Padding
  Conditioning
  Video
  Audio
}

///|
pub(all) suberror PlanError {
  InvalidShape
  InvalidRows
  ElementLimitExceeded
}

///|
pub struct Row {
  source : Source
  index : Int
}

///|
/// Immutable destination-row mapping. Unassigned rows are zero padding, not
/// hidden conditioning. Each supplied source row appears exactly once.
pub struct Plan {
  priv rows : ReadOnlyArray[Row]
  priv width : Int
  priv conditioning_rows : Int
  priv video_rows : Int
  priv audio_rows : Int
  priv mapping : Bytes
}

///|
pub fn Plan::new(
  width~ : Int,
  total_rows~ : Int,
  conditioning~ : ArrayView[Int],
  video~ : ArrayView[Int],
  audio~ : ArrayView[Int],
  max_elements~ : Int64,
) -> Self raise PlanError {
  guard width > 0 && total_rows > 0 && total_rows <= 268435455 else {
    raise InvalidShape
  }
  guard max_elements > 0L &&
    total_rows.to_int64() <= max_elements / width.to_int64() else {
    raise ElementLimitExceeded
  }
  let rows : Array[Row] = Array::make(total_rows, { source: Padding, index: 0, })
  let mapping = FixedArray::make(total_rows * 8, b'\x00')
  for
    (source, tag, positions) in [
      (Conditioning, 1U, conditioning),
      (Video, 2U, video),
      (Audio, 3U, audio),
    ] {
    for index, position in positions {
      guard position >= 0 && position < total_rows else { raise InvalidRows }
      guard rows[position].source is Padding else { raise InvalidRows }
      rows[position] = { source, index, }
      for byte in 0..<4 {
        mapping[position * 8 + byte] = (tag >> (byte * 8)).to_byte()
        mapping[position * 8 + 4 + byte] = (index.reinterpret_as_uint() >>
        (byte * 8)).to_byte()
      }
    }
  }
  {
    rows: ReadOnlyArray::from_array(rows),
    width,
    mapping: Bytes::from_array(mapping),
    conditioning_rows: conditioning.length(),
    video_rows: video.length(),
    audio_rows: audio.length(),
  }
}

///|
pub fn Plan::width(self : Self) -> Int {
  self.width
}

///|
pub fn Plan::rows(self : Self) -> ReadOnlyArray[Row] {
  self.rows
}

///|
pub fn Plan::element_count(self : Self) -> Int64 {
  self.rows.length().to_int64() * self.width.to_int64()
}

///|
pub fn Plan::mapping(self : Self) -> Bytes {
  self.mapping
}

///|
pub fn Plan::input_rows(self : Self, source : Source) -> Int {
  match source {
    Padding => 0
    Conditioning => self.conditioning_rows
    Video => self.video_rows
    Audio => self.audio_rows
  }
}
