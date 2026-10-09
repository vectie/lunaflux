///|
pub(all) suberror ScheduleError {
  InvalidStepCount
  InvalidVisionOverride
  IndexBudgetExceeded
}

///|
pub struct TimestepPattern {
  indices : ReadOnlyArray[Int]
  combined_adaln : ReadOnlyArray[Int]
}

///|
pub struct ScheduledTime {
  values : ReadOnlyArray[Float]
  pattern : Int
}

///|
pub struct TimestepSchedule {
  priv steps : ReadOnlyArray[ScheduledTime]
  priv patterns : ReadOnlyArray[TimestepPattern]
}

///|
/// Prepare the complete schedule once. Only a new F32 equality/order pattern
/// expands row metadata; other evaluations share the immutable pattern. The
/// byte budget covers two Int32 index tables per pattern, not host overhead.
/// Vision overrides refer only to text-segment rows (FL2VA image spans).
pub fn Sequence::schedule(
  self : Self,
  video~ : ArrayView[Double],
  audio~ : ArrayView[Double],
  visual_noise~ : Double,
  audio_noise~ : Double,
  vision_text_rows~ : ArrayView[Int],
  max_steps~ : Int,
  max_index_bytes~ : Int64,
) -> TimestepSchedule raise {
  guard video.length() > 0 &&
    video.length() == audio.length() &&
    video.length() <= max_steps else {
    raise InvalidStepCount
  }
  let rows = self.assembly.rows().length()
  let tags = Array::make(rows, 0)
  for segment in self.segments {
    let tag = match segment.kind {
      Text => 1
      ConditionAudio | TargetAudio => 2
      ConditionVideo | TargetVideo => 0
    }
    for row in segment.start..<(segment.start + segment.count) {
      tags[row] = tag
    }
  }
  for row in vision_text_rows {
    guard row >= 0 && row < self.segments[0].count else {
      raise InvalidVisionOverride
    }
    tags[row] = 0
  }
  let patterns : Array[TimestepPattern] = []
  let steps : Array[ScheduledTime] = []
  let known : Map[String, Int] = Map([])
  let bytes_per_pattern = rows.to_int64() * 8L
  for step in 0..<video.length() {
    let selection = self.select_times(
      video=video[step],
      audio=audio[step],
      visual_noise~,
      audio_noise~,
    )
    let key = "\{selection.video}:\{selection.audio}:\{selection.visual}:\{selection.reference}"
    let pattern = match known.get(key) {
      Some(index) => index
      None => {
        guard bytes_per_pattern <= max_index_bytes &&
          patterns.length().to_int64() < max_index_bytes / bytes_per_pattern else {
          raise IndexBudgetExceeded
        }
        let indices = self.expand_times(selection)
        let combined = Array::makei(rows, row => indices[row] * 3 + tags[row])
        let index = patterns.length()
        patterns.push({
          indices,
          combined_adaln: ReadOnlyArray::from_array(combined),
        })
        known[key] = index
        index
      }
    }
    steps.push({ values: selection.values, pattern, })
  }
  {
    steps: ReadOnlyArray::from_array(steps),
    patterns: ReadOnlyArray::from_array(patterns),
  }
}

///|
pub fn TimestepSchedule::steps(self : Self) -> ReadOnlyArray[ScheduledTime] {
  self.steps
}

///|
pub fn TimestepSchedule::patterns(
  self : Self,
) -> ReadOnlyArray[TimestepPattern] {
  self.patterns
}
