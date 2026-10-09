///|
fn checked_product(
  values : ArrayView[Int64],
) -> Int64 raise JointConditioningReferenceError {
  let mut product = 1L
  for value in values {
    if value <= 0L || product > 9223372036854775807L / value {
      raise GeometryOverflow
    }
    product *= value
  }
  product
}

///|
fn gap_tag(gap : JointConditioningAotGap) -> Int {
  match gap {
    DecodedMediaPayloadBuffers => 1
    VideoVaeEncodeExecution => 2
    VideoVaePosteriorSampling => 3
    VideoVaeLatentNormalization => 4
    AudioVaeEncodeExecution => 5
    AudioVaeLatentNormalization => 6
    ConditioningNoiseValues => 7
    ConditioningScaleNoiseArithmetic => 8
    PackedConditioningRowSpans => 9
    PerReferenceEncodedLatentGeometry => 10
    ReferenceVideoSoundtrackAssociation => 11
  }
}

///|
fn operation_text(
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
) -> String {
  match operation {
    OrderedKeyframeConditioning(maximum, channels) =>
      "keyframes:" + maximum.to_string() + ":" + channels.to_string()
    OrderedReferenceConditioning(images, videos, audio, total, audio_only) =>
      "references:" +
      images.to_string() +
      ":" +
      videos.to_string() +
      ":" +
      audio.to_string() +
      ":" +
      total.to_string() +
      ":" +
      audio_only.to_string()
    _ => "unsupported"
  }
}

///|
fn workflow_tag(workflow : @joint_diffusion_plan.JointDiffusionWorkflow) -> Int {
  match workflow {
    TextToAudioVideo => 1
    FirstLastFrameToAudioVideo => 2
    ReferenceToAudioVideo => 3
  }
}

///|
fn optional_int_text(value : Int?) -> String {
  match value {
    Some(number) => number.to_string()
    None => "none"
  }
}

///|
fn optional_int64_text(value : Int64?) -> String {
  match value {
    Some(number) => number.to_string()
    None => "none"
  }
}

///|
fn conditioning_gap_digest(
  geometry : JointConditioningGeometry,
  gaps : ReadOnlyArray[JointConditioningAotGap],
) -> JointConditioningAotGapDigest {
  let out = StringBuilder()
  out.write_string("lunaflux.joint-conditioning-aot-gap.v1|content=")
  out.write_string(geometry.source_identity.content().as_hex())
  out.write_string("|plan=")
  out.write_string(geometry.source_identity.plan().as_hex())
  out.write_string("|requirements=")
  out.write_string(geometry.requirements_digest.as_hex())
  out.write_string("|request=")
  out.write_string(geometry.request_digest.as_hex())
  out.write_string("|ordinal=")
  out.write_string(geometry.requirement_ordinal.to_string())
  out.write_string("|operation=")
  out.write_string(operation_text(geometry.operation))
  out.write_string("|workflow=")
  out.write_string(workflow_tag(geometry.workflow).to_string())
  out.write_string("|counts=")
  out.write_string(geometry.keyframe_count.to_string())
  out.write_char(':')
  out.write_string(geometry.image_reference_count.to_string())
  out.write_char(':')
  out.write_string(geometry.video_reference_count.to_string())
  out.write_char(':')
  out.write_string(geometry.audio_reference_count.to_string())
  out.write_string("|visual=")
  out.write_string(geometry.visual_latent_channels.to_string())
  out.write_char(':')
  out.write_string(optional_int_text(geometry.visual_row_width))
  out.write_char(':')
  out.write_string(optional_int64_text(geometry.visual_rows_per_condition))
  out.write_char(':')
  out.write_string(optional_int64_text(geometry.total_visual_conditioning_rows))
  out.write_char(':')
  out.write_string(
    optional_int64_text(geometry.total_visual_conditioning_elements),
  )
  for gap in gaps {
    out.write_string("|gap=")
    out.write_string(gap_tag(gap).to_string())
  }
  {
    hex: @crypto.bytes_to_hex_string(
      @crypto.sha256(@utf8.encode(out.to_string())),
    ),
  }
}

///|
fn validate_request_shape(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_io_plan.JointDiffusionIoPlan,
) -> Unit raise JointConditioningReferenceError {
  if request.identity() != requirements.source_identity() {
    raise InvalidRequest
  }
  let shape = request.shape()
  let reconstructed = requirements
    .shape_contract()
    .resolve_request(
      height=shape.height(),
      width=shape.width(),
      requested_frames=shape.frames(),
      inference_steps=shape.inference_steps(),
    ) catch {
      _ => raise InvalidRequest
    }
  if reconstructed != shape {
    raise InvalidRequest
  }
}

///|
fn keyframe_geometry(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  requirement_ordinal : Int,
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
  maximum : Int,
  channels : Int,
  limits : JointConditioningReferenceLimits,
) -> JointConditioningAotAssessment raise JointConditioningReferenceError {
  let count = request.keyframes().length()
  if request.workflow() != FirstLastFrameToAudioVideo ||
    count <= 0 ||
    count > maximum ||
    channels != request.video_latent_channels() ||
    request.references().length() != 0 {
    raise InvalidRequest
  }
  let shape = request.shape()
  let patch_t = request.video_patch_temporal()
  let patch_h = request.video_patch_height()
  let patch_w = request.video_patch_width()
  if patch_t != 1 ||
    shape.video_latent_height() % patch_h != 0 ||
    shape.video_latent_width() % patch_w != 0 {
    raise InvalidRequirements
  }
  let row_width64 = checked_product([
    channels.to_int64(),
    patch_t.to_int64(),
    patch_h.to_int64(),
    patch_w.to_int64(),
  ])
  if row_width64 > 2147483647L {
    raise GeometryOverflow
  }
  let rows_per_condition = checked_product([
    (1 / patch_t).to_int64(),
    (shape.video_latent_height() / patch_h).to_int64(),
    (shape.video_latent_width() / patch_w).to_int64(),
  ])
  let total_rows = checked_product([count.to_int64(), rows_per_condition])
  let total_elements = checked_product([total_rows, row_width64])
  if total_rows > limits.maximum_conditioning_rows ||
    total_elements > limits.maximum_conditioning_elements {
    raise GeometryBudgetExceeded
  }
  let geometry : JointConditioningGeometry = {
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    request_digest: request.request_digest(),
    requirement_ordinal,
    operation,
    workflow: request.workflow(),
    keyframe_count: count,
    image_reference_count: 0,
    video_reference_count: 0,
    audio_reference_count: 0,
    visual_latent_channels: channels,
    visual_row_width: Some(row_width64.to_int()),
    visual_rows_per_condition: Some(rows_per_condition),
    total_visual_conditioning_rows: Some(total_rows),
    total_visual_conditioning_elements: Some(total_elements),
  }
  let gaps : ReadOnlyArray[JointConditioningAotGap] = [
    DecodedMediaPayloadBuffers,
    VideoVaeEncodeExecution,
    VideoVaePosteriorSampling,
    VideoVaeLatentNormalization,
    ConditioningNoiseValues,
    ConditioningScaleNoiseArithmetic,
    PackedConditioningRowSpans,
  ]
  { geometry, gap_digest: conditioning_gap_digest(geometry, gaps), gaps, }
}

///|
fn reference_geometry(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  requirement_ordinal : Int,
  operation : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
  maximum_images : Int,
  maximum_videos : Int,
  maximum_audio : Int,
  maximum_total : Int,
  allow_audio_only : Bool,
) -> JointConditioningAotAssessment raise JointConditioningReferenceError {
  if request.workflow() != ReferenceToAudioVideo ||
    request.keyframes().length() != 0 ||
    request.references().is_empty() ||
    request.references().length() > maximum_total {
    raise InvalidRequest
  }
  let mut images = 0
  let mut videos = 0
  let mut audio = 0
  for reference in request.references() {
    match reference {
      ImageReference(_) => images += 1
      VideoReference(_) => videos += 1
      AudioReference(_) => audio += 1
    }
  }
  if images > maximum_images ||
    videos > maximum_videos ||
    audio > maximum_audio ||
    (!allow_audio_only && images == 0 && videos == 0) {
    raise InvalidRequest
  }
  let geometry : JointConditioningGeometry = {
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    request_digest: request.request_digest(),
    requirement_ordinal,
    operation,
    workflow: request.workflow(),
    keyframe_count: 0,
    image_reference_count: images,
    video_reference_count: videos,
    audio_reference_count: audio,
    visual_latent_channels: request.video_latent_channels(),
    visual_row_width: None,
    visual_rows_per_condition: None,
    total_visual_conditioning_rows: None,
    total_visual_conditioning_elements: None,
  }
  let gaps : Array[JointConditioningAotGap] = [
    DecodedMediaPayloadBuffers,
    VideoVaeEncodeExecution,
    VideoVaePosteriorSampling,
    VideoVaeLatentNormalization,
  ]
  if audio > 0 {
    gaps.push(AudioVaeEncodeExecution)
    gaps.push(AudioVaeLatentNormalization)
  }
  gaps.push(ConditioningNoiseValues)
  gaps.push(ConditioningScaleNoiseArithmetic)
  gaps.push(PackedConditioningRowSpans)
  gaps.push(PerReferenceEncodedLatentGeometry)
  if videos > 0 {
    gaps.push(ReferenceVideoSoundtrackAssociation)
  }
  let gaps = ReadOnlyArray::from_array(gaps)
  { geometry, gap_digest: conditioning_gap_digest(geometry, gaps), gaps, }
}

///|
/// Resolves only ordered conditioning geometry already authenticated by the
/// request and requirements. It deliberately produces no executable candidate.
pub fn assess_ordered_conditioning_aot(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  request : @joint_diffusion_io_plan.JointDiffusionIoPlan,
  requirement_ordinal~ : Int,
  limits~ : JointConditioningReferenceLimits,
) -> JointConditioningAotAssessment raise JointConditioningReferenceError {
  validate_request_shape(requirements, request)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  match (requirement.operation(), requirement.capability()) {
    (
      OrderedKeyframeConditioning(maximum, channels),
      OrderedKeyframeConditioningAot,
    ) =>
      keyframe_geometry(
        requirements,
        request,
        requirement_ordinal,
        OrderedKeyframeConditioning(maximum, channels),
        maximum,
        channels,
        limits,
      )
    (
      OrderedReferenceConditioning(images, videos, audio, total, audio_only),
      OrderedReferenceConditioningAot,
    ) =>
      reference_geometry(
        requirements,
        request,
        requirement_ordinal,
        OrderedReferenceConditioning(images, videos, audio, total, audio_only),
        images,
        videos,
        audio,
        total,
        audio_only,
      )
    _ => raise UnsupportedRequirement(requirement_ordinal)
  }
}

///|
/// Fails closed because the assessment intentionally carries no artifact or
/// execution authority.
pub fn require_ordered_conditioning_aot_candidate(
  assessment : JointConditioningAotAssessment,
) -> Unit raise JointConditioningReferenceError {
  guard assessment.gaps.get(0) is Some(first) else { raise InvalidRequirements }
  raise ConditioningSemanticUnavailable(assessment.geometry.operation, first)
}
