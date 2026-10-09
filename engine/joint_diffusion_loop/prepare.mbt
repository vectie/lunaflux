///|
pub(all) suberror PreparationError {
  InvalidRequest
  InvalidScheduler
  ElementLimitExceeded
} derive(Debug, Eq)

///|
pub extend PreparationError with @debug.Debug::{to_repr}

///|
pub extend PreparationError with Eq::{equal, not_equal}

///|
/// Pure request-bound plan. Counts describe raw latents, not projected rows or
/// VAE decoder batches. Storage precision and scratch belong to device plans.
pub struct PreparedPlan {
  priv model_digest : @spec.PlanDigest
  priv request : @joint_diffusion_plan.JointDiffusionRequestShape
  priv steps : ReadOnlyArray[JointStep]
  priv video_elements : Int64
  priv audio_elements : Int64
  priv video_pack : @dense_permutation_plan.Permutation
  priv audio_unpack : @dense_permutation_plan.Permutation
  priv flow_table : FlowTable
}

///|
fn bounded_product(
  values : ArrayView[Int],
  ceiling : Int64,
) -> Int64 raise PreparationError {
  for value in values; product = 1L {
    guard value > 0 && product <= ceiling / value.to_int64() else {
      raise ElementLimitExceeded
    }
    continue product * value.to_int64()
  } nobreak {
    product
  }
}

///|
fn sigma_at(
  scheduler : @joint_diffusion_plan.JointDiffusionScheduler,
  count : Int,
  index : Int,
) -> Double {
  if index == count {
    return 0.0
  }
  let fraction = index.to_double() / (count - 1).to_double()
  let base = 1.0 +
    (1.0 / scheduler.training_timestep_count().to_double() - 1.0) * fraction
  let shift = scheduler.shift()
  shift * base / (1.0 + (shift - 1.0) * base)
}

///|
/// Prepare once from family-neutral semantics, before allocating tensors.
/// The independent reference implementation remains test-only.
pub fn PreparedPlan::new(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  max_latent_elements~ : Int64,
) -> PreparedPlan raise PreparationError {
  let shapes = model.shapes()
  let reconstructed = shapes.resolve_request(
    height=request.height(),
    width=request.width(),
    requested_frames=request.frames(),
    inference_steps=request.inference_steps(),
  ) catch {
    _ => raise InvalidRequest
  }
  guard reconstructed == request && model.schema_version() == 1 else {
    raise InvalidRequest
  }
  let video = model.video_scheduler()
  let audio = model.audio_scheduler()
  guard video.latent() == VideoLatent &&
    audio.latent() == AudioLatent &&
    video.kind() == RectifiedFlowEuler &&
    audio.kind() == RectifiedFlowEuler else {
    raise InvalidScheduler
  }
  guard max_latent_elements > 0L else { raise ElementLimitExceeded }
  let video_elements = bounded_product(
    [
      shapes.video_latent_channels(),
      request.video_latent_frames(),
      request.video_latent_height(),
      request.video_latent_width(),
    ],
    max_latent_elements,
  )
  let audio_elements = bounded_product(
    [shapes.audio_latent_channels(), request.audio_token_rows()],
    max_latent_elements - video_elements,
  )
  let pt = shapes.video_patch_temporal()
  let ph = shapes.video_patch_height()
  let pw = shapes.video_patch_width()
  guard request.video_latent_frames() % pt == 0 &&
    request.video_latent_height() % ph == 0 &&
    request.video_latent_width() % pw == 0 else {
    raise InvalidRequest
  }
  // Retain patch-major state throughout denoising. Only the input/output
  // boundaries transpose; every per-element RF update keeps this ordering.
  let video_pack = @dense_permutation_plan.Permutation::new(
    [
      1,
      shapes.video_latent_channels(),
      request.video_latent_frames() / pt,
      pt,
      request.video_latent_height() / ph,
      ph,
      request.video_latent_width() / pw,
      pw,
    ],
    [0, 2, 4, 6, 1, 3, 5, 7],
    max_elements=video_elements,
  ) catch {
    _ => raise InvalidRequest
  }
  let audio_unpack = @dense_permutation_plan.Permutation::new(
    [
      shapes.audio_latent_streams(),
      request.audio_latent_frames(),
      shapes.audio_latent_channels(),
    ],
    [0, 2, 1],
    max_elements=audio_elements,
  ) catch {
    _ => raise InvalidRequest
  }
  let count = request.model_evaluations()
  guard count >= 2 && count == request.inference_steps() else {
    raise InvalidRequest
  }
  let steps : Array[JointStep] = []
  for i in 0..<count {
    let vs = sigma_at(video, count, i)
    let as_ = sigma_at(audio, count, i)
    steps.push({
      video_timestep: 1.0 - vs,
      audio_timestep: 1.0 - as_,
      video_sigma: vs,
      audio_sigma: as_,
      video_next_sigma: sigma_at(video, count, i + 1),
      audio_next_sigma: sigma_at(audio, count, i + 1),
    })
  }
  // Validate scalar records once, never in the execution loop.
  let validated = Loop::new(steps) catch { _ => raise InvalidScheduler }
  {
    model_digest: model.plan_digest(),
    request,
    steps: validated.steps,
    video_elements,
    audio_elements,
    video_pack,
    audio_unpack,
    flow_table: build_flow_table(validated.steps),
  }
}

///|
pub fn PreparedPlan::start(self : Self) -> Loop {
  { steps: self.steps, completed: 0, progress: Running, executing: false, }
}

///|
pub fn PreparedPlan::steps(self : Self) -> ReadOnlyArray[JointStep] {
  self.steps
}

///|
pub fn PreparedPlan::model_digest(self : Self) -> @spec.PlanDigest {
  self.model_digest
}

///|
pub fn PreparedPlan::request(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  self.request
}

///|
pub fn PreparedPlan::video_elements(self : Self) -> Int64 {
  self.video_elements
}

///|
pub fn PreparedPlan::audio_elements(self : Self) -> Int64 {
  self.audio_elements
}

///|
/// Shared immutable payload prepared once, not re-encoded on access.
pub fn PreparedPlan::flow_table(self : Self) -> FlowTable {
  self.flow_table
}

///|
/// Raw [1,C,T,H,W] to [t*h*w,C*pt*ph*pw]. Use inverse() for VAE input.
pub fn PreparedPlan::video_pack(
  self : Self,
) -> @dense_permutation_plan.Permutation {
  self.video_pack
}

///|
/// Stream-major [S*T,C] denoiser rows to [S,C,T] AudioVAE input.
pub fn PreparedPlan::audio_unpack(
  self : Self,
) -> @dense_permutation_plan.Permutation {
  self.audio_unpack
}
