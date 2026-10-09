///|
fn request_matches_plan(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
) -> Bool {
  let reconstructed = plan
    .shapes()
    .resolve_request(
      height=request.height(),
      width=request.width(),
      requested_frames=request.frames(),
      inference_steps=request.inference_steps(),
    ) catch {
      _ => return false
    }
  reconstructed == request
}

///|
fn scheduler_for(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> @joint_diffusion_plan.JointDiffusionScheduler {
  match latent {
    VideoLatent => plan.video_scheduler()
    AudioLatent => plan.audio_scheduler()
  }
}

///|
/// Builds the normalized shifted Rectified Flow reference grid:
///
/// `shifted = shift * base / (1 + (shift - 1) * base)`
///
/// The inference grid spans training sigma `1` through `1 / T`, applies the
/// shift, and appends terminal sigma `0`. Thus `N` model timesteps have `N + 1`
/// sigma points. Timesteps use the scheduler's plan-bound training count.
pub fn build_shifted_rectified_flow_schedule(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  latent : @joint_diffusion_plan.JointDiffusionLatentKind,
) -> ShiftedRectifiedFlowSchedule raise JointDiffusionReferenceError {
  if plan.schema_version() != 1 || !request_matches_plan(plan, request) {
    raise InvalidRequestShape
  }
  let scheduler = scheduler_for(plan, latent)
  if scheduler.latent() != latent || scheduler.kind() != RectifiedFlowEuler {
    raise InvalidPlanContract
  }
  let evaluation_count = request.inference_steps()
  if evaluation_count < 2 || request.model_evaluations() != evaluation_count {
    raise InvalidSchedule
  }
  let shift = scheduler.shift()
  if shift <= 0.0 || shift.is_nan() || shift.is_inf() {
    raise InvalidSchedule
  }
  let denominator = (evaluation_count - 1).to_double()
  let minimum_sigma = 1.0 / scheduler.training_timestep_count().to_double()
  let sigmas : Array[Double] = []
  let timesteps : Array[Double] = []
  for index in 0..<evaluation_count {
    let fraction = index.to_double() / denominator
    let base = 1.0 + (minimum_sigma - 1.0) * fraction
    let shifted = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas.push(shifted)
    timesteps.push(shifted * scheduler.training_timestep_count().to_double())
  }
  sigmas.push(0.0)
  if sigmas[0] != 1.0 || sigmas[evaluation_count] != 0.0 {
    raise InvalidSchedule
  }
  {
    latent,
    plan_digest: plan.plan_digest(),
    sigmas: ReadOnlyArray::from_array(sigmas),
    timesteps: ReadOnlyArray::from_array(timesteps),
  }
}

///|
pub fn ShiftedRectifiedFlowSchedule::latent(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionLatentKind {
  self.latent
}

///|
pub fn ShiftedRectifiedFlowSchedule::plan_digest(
  self : Self,
) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn ShiftedRectifiedFlowSchedule::point_count(self : Self) -> Int {
  self.sigmas.length()
}

///|
pub fn ShiftedRectifiedFlowSchedule::evaluation_count(self : Self) -> Int {
  self.timesteps.length()
}

///|
pub fn ShiftedRectifiedFlowSchedule::sigma(self : Self, index : Int) -> Double? {
  self.sigmas.get(index)
}

///|
pub fn ShiftedRectifiedFlowSchedule::timestep(
  self : Self,
  index : Int,
) -> Double? {
  self.timesteps.get(index)
}
