///|
/// Resolves the canonical projected transformer row order. Video rows are
/// patch-major `(time, height, width)` rows; audio rows follow in temporal
/// order. Conditioning rows always precede both latent modalities.
pub fn JointPackedRowLayout::new(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  conditioning_rows~ : Int,
) -> JointPackedRowLayout raise JointDiffusionReferenceError {
  if plan.schema_version() != 1 ||
    !plan.packed_joint_sequence() ||
    !request_matches_plan(plan, request) {
    raise InvalidPlanContract
  }
  if conditioning_rows < 0 || conditioning_rows > 1000000 {
    raise InvalidConditioningRows
  }
  let shapes = plan.shapes()
  let patch_t = shapes.video_patch_temporal()
  let patch_h = shapes.video_patch_height()
  let patch_w = shapes.video_patch_width()
  if request.video_latent_frames() % patch_t != 0 ||
    request.video_latent_height() % patch_h != 0 ||
    request.video_latent_width() % patch_w != 0 {
    raise InvalidRequestShape
  }
  let video_rows = exact_int(
    checked_product([
      (request.video_latent_frames() / patch_t).to_int64(),
      (request.video_latent_height() / patch_h).to_int64(),
      (request.video_latent_width() / patch_w).to_int64(),
    ]),
  )
  let audio_rows = request.audio_token_rows()
  let video_input_width = exact_int(
    checked_product([
      shapes.video_latent_channels().to_int64(),
      patch_t.to_int64(),
      patch_h.to_int64(),
      patch_w.to_int64(),
    ]),
  )
  let total_rows = conditioning_rows.to_int64() +
    video_rows.to_int64() +
    audio_rows.to_int64()
  ignore(exact_int(total_rows))
  {
    plan_digest: plan.plan_digest(),
    hidden_width: plan.denoiser().hidden_size(),
    conditioning_rows,
    video_rows,
    audio_rows,
    video_input_width,
    audio_input_width: shapes.audio_latent_channels(),
  }
}

///|
fn expected_projected_elements(
  layout : JointPackedRowLayout,
  segment : JointPackedRowSegment,
) -> Int raise JointDiffusionReferenceError {
  let rows = match segment {
    ConditioningRows => layout.conditioning_rows
    VideoRows => layout.video_rows
    AudioRows => layout.audio_rows
  }
  if rows == 0 {
    return 0
  }
  exact_int(checked_product([rows.to_int64(), layout.hidden_width.to_int64()]))
}

///|
/// Concatenates already-projected rows in the plan-required order. This
/// bounded reference copy is intended for fixtures and correctness oracles.
pub fn recombine_projected_rows(
  layout : JointPackedRowLayout,
  conditioning : ArrayView[Double],
  video : ArrayView[Double],
  audio : ArrayView[Double],
  max_elements~ : Int,
) -> JointProjectedRows raise JointDiffusionReferenceError {
  let conditioning_expected = expected_projected_elements(
    layout,
    ConditioningRows,
  )
  let video_expected = expected_projected_elements(layout, VideoRows)
  let audio_expected = expected_projected_elements(layout, AudioRows)
  if conditioning.length() != conditioning_expected {
    raise InvalidProjectedRowCount(ConditioningRows)
  }
  if video.length() != video_expected {
    raise InvalidProjectedRowCount(VideoRows)
  }
  if audio.length() != audio_expected {
    raise InvalidProjectedRowCount(AudioRows)
  }
  let total = conditioning_expected + video_expected + audio_expected
  if max_elements <= 0 || total > max_elements {
    raise PackedElementLimitExceeded
  }
  let values : Array[Double] = []
  for value in conditioning {
    values.push(value)
  }
  for value in video {
    values.push(value)
  }
  for value in audio {
    values.push(value)
  }
  { layout, values: ReadOnlyArray::from_array(values), }
}

///|
pub fn split_projected_rows(
  packed : JointProjectedRows,
) -> JointProjectedRowSplit {
  let conditioning_end = try! expected_projected_elements(
    packed.layout,
    ConditioningRows,
  )
  let video_elements = try! expected_projected_elements(
    packed.layout,
    VideoRows,
  )
  let video_end = conditioning_end + video_elements
  let conditioning : Array[Double] = []
  let video : Array[Double] = []
  let audio : Array[Double] = []
  for index, value in packed.values {
    if index < conditioning_end {
      conditioning.push(value)
    } else if index < video_end {
      video.push(value)
    } else {
      audio.push(value)
    }
  }
  {
    conditioning: ReadOnlyArray::from_array(conditioning),
    video: ReadOnlyArray::from_array(video),
    audio: ReadOnlyArray::from_array(audio),
  }
}

///|
pub fn JointPackedRowLayout::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointPackedRowLayout::hidden_width(self : Self) -> Int {
  self.hidden_width
}

///|
pub fn JointPackedRowLayout::conditioning_rows(self : Self) -> Int {
  self.conditioning_rows
}

///|
pub fn JointPackedRowLayout::video_rows(self : Self) -> Int {
  self.video_rows
}

///|
pub fn JointPackedRowLayout::audio_rows(self : Self) -> Int {
  self.audio_rows
}

///|
pub fn JointPackedRowLayout::total_rows(self : Self) -> Int {
  self.conditioning_rows + self.video_rows + self.audio_rows
}

///|
pub fn JointPackedRowLayout::row_start(
  self : Self,
  segment : JointPackedRowSegment,
) -> Int {
  match segment {
    ConditioningRows => 0
    VideoRows => self.conditioning_rows
    AudioRows => self.conditioning_rows + self.video_rows
  }
}

///|
pub fn JointPackedRowLayout::row_count(
  self : Self,
  segment : JointPackedRowSegment,
) -> Int {
  match segment {
    ConditioningRows => self.conditioning_rows
    VideoRows => self.video_rows
    AudioRows => self.audio_rows
  }
}

///|
pub fn JointPackedRowLayout::video_input_width(self : Self) -> Int {
  self.video_input_width
}

///|
pub fn JointPackedRowLayout::audio_input_width(self : Self) -> Int {
  self.audio_input_width
}

///|
pub fn JointProjectedRows::layout(self : Self) -> JointPackedRowLayout {
  self.layout
}

///|
pub fn JointProjectedRows::values(self : Self) -> ReadOnlyArray[Double] {
  self.values
}

///|
pub fn JointProjectedRowSplit::conditioning(
  self : Self,
) -> ReadOnlyArray[Double] {
  self.conditioning
}

///|
pub fn JointProjectedRowSplit::video(self : Self) -> ReadOnlyArray[Double] {
  self.video
}

///|
pub fn JointProjectedRowSplit::audio(self : Self) -> ReadOnlyArray[Double] {
  self.audio
}
