///|
pub(all) suberror TimestepError {
  InvalidTime
}

///|
pub struct Timesteps {
  priv values : ReadOnlyArray[Float]
  priv indices : ReadOnlyArray[Int]
}

///|
priv struct TimeSelection {
  values : ReadOnlyArray[Float]
  video : Int
  audio : Int
  visual : Int
  reference : Int
}

///|
/// Startup-only per-evaluation metadata. Text/padding inherit video time;
/// visual/audio conditions use max(target time, noise-augmentation time).
/// Deduplicate after F32 conversion, in ascending order, matching H3's
/// candidate torch.unique semantics. No conditioning encoder is run here.
pub fn Sequence::timesteps(
  self : Self,
  video~ : Double,
  audio~ : Double,
  visual_noise~ : Double,
  audio_noise~ : Double,
) -> Timesteps raise TimestepError {
  let selection = self.select_times(video~, audio~, visual_noise~, audio_noise~)
  { values: selection.values, indices: self.expand_times(selection), }
}

///|
fn Sequence::select_times(
  self : Self,
  video~ : Double,
  audio~ : Double,
  visual_noise~ : Double,
  audio_noise~ : Double,
) -> TimeSelection raise TimestepError {
  for value in [video, audio, visual_noise, audio_noise] {
    guard !value.is_nan() && !value.is_inf() && value >= 0.0 && value <= 1.0 else {
      raise InvalidTime
    }
  }
  let video_time = Float::from_double(video)
  let audio_time = Float::from_double(audio)
  let visual_time = Float::from_double(
    if video > visual_noise {
      video
    } else {
      visual_noise
    },
  )
  let reference_time = Float::from_double(
    if audio > audio_noise {
      audio
    } else {
      audio_noise
    },
  )
  let has_visual = self.segments.any(segment => segment.kind is ConditionVideo)
  let has_reference = self.segments.any(segment => {
    segment.kind is ConditionAudio
  })
  let candidates = [video_time, audio_time]
  if has_visual {
    candidates.push(visual_time)
  }
  if has_reference {
    candidates.push(reference_time)
  }
  let values : Array[Float] = []
  // At most four values regardless of sequence length or reference count.
  for value in candidates {
    if !values.contains(value) {
      values.push(value)
    }
  }
  values.sort()
  fn index(value : Float) -> Int {
    for i, item in values {
      if item == value {
        return i
      }
    }
    abort("prepared timestep missing from candidate set")
  }
  {
    values: ReadOnlyArray::from_array(values),
    video: index(video_time),
    audio: index(audio_time),
    visual: if has_visual {
      index(visual_time)
    } else {
      -1
    },
    reference: if has_reference {
      index(reference_time)
    } else {
      -1
    },
  }
}

///|
fn Sequence::expand_times(
  self : Self,
  selection : TimeSelection,
) -> ReadOnlyArray[Int] {
  let indices = Array::make(self.assembly.rows().length(), selection.video)
  for segment in self.segments {
    let selected = match segment.kind {
      Text | TargetVideo => selection.video
      TargetAudio => selection.audio
      ConditionVideo => selection.visual
      ConditionAudio => selection.reference
    }
    for row in segment.start..<(segment.start + segment.count) {
      indices[row] = selected
    }
  }
  ReadOnlyArray::from_array(indices)
}

///|
pub fn Timesteps::values(self : Self) -> ReadOnlyArray[Float] {
  self.values
}

///|
pub fn Timesteps::indices(self : Self) -> ReadOnlyArray[Int] {
  self.indices
}
