///|
pub(all) enum KeyframeAnchor {
  First
  Last
}

///|
fn frame_span(index : Int) -> Double {
  5.0 / 3.0 * (if index % 5 == 0 { 1.0 } else { 4.0 })
}

///|
/// FL2VA uses an eight-lane, 128-element FP64 pairwise reduction, unlike
/// Ref2VA's sequential span. Keep this numerical policy distinct.
fn keyframe_span(start : Int, count : Int) -> Double {
  if count < 8 {
    for i in 0..<count; sum = -0.0 {
      continue sum + frame_span(start + i)
    } nobreak {
      sum
    }
  } else if count <= 128 {
    let lanes = FixedArray::makei(8, lane => frame_span(start + lane))
    let aligned = count - count % 8
    for base = 8; base < aligned; base = base + 8 {
      for lane in 0..<8 {
        lanes[lane] = lanes[lane] + frame_span(start + base + lane)
      }
    }
    let left = lanes[0] + lanes[1] + (lanes[2] + lanes[3])
    let right = lanes[4] + lanes[5] + (lanes[6] + lanes[7])
    for i in aligned..<count; sum = left + right {
      continue sum + frame_span(start + i)
    } nobreak {
      sum
    }
  } else {
    let half = count / 2
    let split = half - half % 8
    keyframe_span(start, split) + keyframe_span(start + split, count - split)
  }
}

///|
/// FL2VA conditions share target spatial dimensions and do not advance target
/// time. Anchors are resolved endpoints, not arbitrary frame indices.
pub fn PositionedSequence::keyframes(
  target : VideoGrid,
  hidden_width~ : Int,
  text_rows~ : Int,
  audio_frames~ : Int,
  anchors~ : ArrayView[KeyframeAnchor],
  max_rows~ : Int,
) -> Self raise SequenceError {
  guard anchors.length() <= 2 else { raise InvalidConditioning }
  guard audio_frames > 0 && audio_frames <= max_rows / 2 else {
    raise InvalidShape
  }
  let mut first = false
  let mut last = false
  let blocks : Array[ConditionBlock] = []
  let frame_rows = target.rows / target.frames
  for anchor in anchors {
    match anchor {
      First => {
        guard !first else { raise InvalidConditioning }
        first = true
      }
      Last => {
        guard !last else { raise InvalidConditioning }
        last = true
      }
    }
    blocks.push(Image(frame_rows))
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
  let frame : VideoGrid = {
    frames: 1,
    height: target.height,
    width: target.width,
    rows: frame_rows,
  }
  let origin = text_rows.to_double()
  let last_time = origin + keyframe_span(0, target.frames) - 5.0 / 3.0
  for anchor in anchors {
    let time = match anchor {
      First => origin
      Last => last_time
    }
    ignore(append_video(positions, frame, time))
  }
  append_audio(positions, audio_frames, target, origin)
  ignore(append_video(positions, target, origin))
  while positions.length() < sequence.assembly.rows().length() {
    positions.push({ time: 0.0, height: 0.0, width: 0.0, })
  }
  { sequence, positions: ReadOnlyArray::from_array(positions), }
}
