///|
fn keyframe_tag(position : JointKeyframePosition) -> Int {
  match position {
    FirstFrame => 1
    LastFrame => 2
  }
}

///|
fn reference_tag(reference : JointReferenceConditioning) -> Int {
  match reference {
    ImageReference(_) => 1
    VideoReference(_) => 2
    AudioReference(_) => 3
  }
}

///|
fn media_digest(reference : JointReferenceConditioning) -> @spec.ContentDigest {
  match reference {
    ImageReference(value) => value.content_digest()
    VideoReference(value) => value.content_digest()
    AudioReference(value) => value.content_digest()
  }
}

///|
fn media_geometry(reference : JointReferenceConditioning) -> String {
  match reference {
    ImageReference(value) => "\{value.height()}:\{value.width()}"
    VideoReference(value) =>
      "\{value.height()}:\{value.width()}:\{value.frame_count()}:\{value.frames_per_second()}"
    AudioReference(value) =>
      "\{value.sample_count()}:\{value.sample_rate()}:\{value.channel_count()}"
  }
}

///|
fn canonical_keyframes(
  keyframes : ArrayView[JointKeyframeConditioning],
) -> ReadOnlyArray[JointKeyframeConditioning] raise JointDiffusionIoPlanError {
  if keyframes.length() == 0 || keyframes.length() > 2 {
    raise InvalidKeyframeConditioning
  }
  let mut first : JointKeyframeConditioning? = None
  let mut last : JointKeyframeConditioning? = None
  for value in keyframes {
    match value.position() {
      FirstFrame =>
        if first is Some(_) {
          raise InvalidKeyframeConditioning
        } else {
          first = Some(value)
        }
      LastFrame =>
        if last is Some(_) {
          raise InvalidKeyframeConditioning
        } else {
          last = Some(value)
        }
    }
  }
  let out : Array[JointKeyframeConditioning] = []
  match first {
    Some(value) => out.push(value)
    None => ()
  }
  match last {
    Some(value) => out.push(value)
    None => ()
  }
  ReadOnlyArray::from_array(out)
}

///|
fn validate_image_canvas(
  image : ValidatedImageInput,
  shape : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Unit raise JointDiffusionIoPlanError {
  if image.height() != shape.height() || image.width() != shape.width() {
    raise InvalidMediaDescriptor
  }
}

///|
fn validate_keyframes(
  keyframes : ReadOnlyArray[JointKeyframeConditioning],
  shape : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Unit raise JointDiffusionIoPlanError {
  for value in keyframes {
    validate_image_canvas(value.image(), shape)
  }
}

///|
fn validate_references(
  references : ReadOnlyArray[JointReferenceConditioning],
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  shape : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Unit raise JointDiffusionIoPlanError {
  let limits = plan.conditioning()
  if references.is_empty() ||
    references.length() > limits.max_total_references() {
    raise InvalidReferenceConditioning
  }
  let mut image_count = 0
  let mut video_count = 0
  let mut audio_count = 0
  for reference in references {
    match reference {
      ImageReference(value) => {
        image_count += 1
        validate_image_canvas(value, shape)
      }
      VideoReference(value) => {
        video_count += 1
        if value.height() != shape.height() ||
          value.width() != shape.width() ||
          value.frames_per_second() != plan.shapes().frames_per_second() ||
          value.frame_count() > shape.frames() {
          raise InvalidMediaDescriptor
        }
      }
      AudioReference(value) => {
        audio_count += 1
        if value.sample_rate() != plan.shapes().audio_sample_rate() ||
          value.channel_count() != plan.shapes().audio_output_channels() ||
          value.sample_count() > shape.audio_samples() {
          raise InvalidMediaDescriptor
        }
      }
    }
  }
  if image_count > limits.max_image_references() ||
    video_count > limits.max_video_references() ||
    audio_count > limits.max_audio_references() ||
    (
      !limits.allow_audio_only_references() &&
      image_count == 0 &&
      video_count == 0
    ) {
    raise InvalidReferenceConditioning
  }
}

///|
fn request_digest(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  workflow : @joint_diffusion_plan.JointDiffusionWorkflow,
  text : ValidatedTextConditioning,
  keyframes : ReadOnlyArray[JointKeyframeConditioning],
  references : ReadOnlyArray[JointReferenceConditioning],
  shape : @joint_diffusion_plan.JointDiffusionRequestShape,
  seed : Int64,
) -> @spec.PlanDigest {
  let out = StringBuilder()
  out <+
    "lunaflux.joint-diffusion-io-plan.v1|model=\{plan.content_digest().as_hex()}:\{plan.plan_digest().as_hex()}"
  out <+
    "|workflow=\{match workflow { TextToAudioVideo => 1; FirstLastFrameToAudioVideo => 2; ReferenceToAudioVideo => 3 }}"
  out <+ "|text=\{text.content_digest().as_hex()}:\{text.token_count()}"
  for value in keyframes {
    let image = value.image()
    out <+
      "|key=\{keyframe_tag(value.position())}:\{image.content_digest().as_hex()}:\{image.height()}:\{image.width()}"
  }
  for value in references {
    out <+
      "|ref=\{reference_tag(value)}:\{media_digest(value).as_hex()}:\{media_geometry(value)}"
  }
  out <+
    "|output=\{shape.height()}:\{shape.width()}:\{shape.frames()}:\{shape.inference_steps()}:\{shape.video_latent_frames()}:\{shape.video_latent_height()}:\{shape.video_latent_width()}:\{shape.audio_latent_frames()}:\{shape.audio_samples()}"
  out <+ "|seed=\{seed}|guidance=distilled-unit|cancel=stage-boundary"
  try! @spec.PlanDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(@utf8.encode(out.to_string()))),
  )
}

///|
pub fn normalize(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  workflow~ : @joint_diffusion_plan.JointDiffusionWorkflow,
  text~ : ValidatedTextConditioning,
  height~ : Int,
  width~ : Int,
  requested_frames~ : Int,
  inference_steps~ : Int,
  seed~ : Int64,
  guidance_scale~ : Double,
  keyframes? : ArrayView[JointKeyframeConditioning] = [],
  references? : ArrayView[JointReferenceConditioning] = [],
) -> JointDiffusionIoPlan raise JointDiffusionIoPlanError {
  if !plan.workflows().contains(workflow) {
    raise WorkflowNotAdmitted
  }
  if guidance_scale.is_nan() ||
    guidance_scale.is_inf() ||
    guidance_scale != 1.0 ||
    !plan.guidance_distilled() {
    raise InvalidGuidancePolicy
  }
  let shape = plan
    .shapes()
    .resolve_request(height~, width~, requested_frames~, inference_steps~) catch {
      _ => raise InvalidOutputShape
    }
  let no_keyframes : ReadOnlyArray[JointKeyframeConditioning] = []
  let no_references : ReadOnlyArray[JointReferenceConditioning] = []
  let (normalized_keyframes, normalized_references) = match workflow {
    TextToAudioVideo => {
      if !keyframes.is_empty() || !references.is_empty() {
        raise InvalidKeyframeConditioning
      }
      (no_keyframes, no_references)
    }
    FirstLastFrameToAudioVideo => {
      if !references.is_empty() ||
        keyframes.length() > plan.conditioning().max_keyframes() {
        raise InvalidKeyframeConditioning
      }
      let normalized = canonical_keyframes(keyframes)
      validate_keyframes(normalized, shape)
      (normalized, no_references)
    }
    ReferenceToAudioVideo => {
      if !keyframes.is_empty() {
        raise InvalidReferenceConditioning
      }
      let normalized = ReadOnlyArray::from_array(references)
      validate_references(normalized, plan, shape)
      (no_keyframes, normalized)
    }
  }
  let identity = try! @spec.ModelIdentity::new(
    plan.content_digest(),
    plan.plan_digest(),
  )
  let digest = request_digest(
    plan, workflow, text, normalized_keyframes, normalized_references, shape, seed,
  )
  {
    schema_version: 1,
    identity,
    request_digest: digest,
    workflow,
    text,
    keyframes: normalized_keyframes,
    references: normalized_references,
    shape,
    video_latent_channels: plan.shapes().video_latent_channels(),
    video_patch_temporal: plan.shapes().video_patch_temporal(),
    video_patch_height: plan.shapes().video_patch_height(),
    video_patch_width: plan.shapes().video_patch_width(),
    audio_latent_channels: plan.shapes().audio_latent_channels(),
    audio_sample_rate: plan.shapes().audio_sample_rate(),
    audio_output_channels: plan.shapes().audio_output_channels(),
    seed,
    guidance: DistilledUnitGuidance,
    cancellation: StageBoundaryCancellation,
    stage_count: plan.stages().length(),
  }
}
