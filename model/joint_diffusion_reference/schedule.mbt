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
/// TrainingFloorFloat64 counts evaluations and appends zero after `1 / T`.
/// InclusiveZeroFloat32 counts points on `[1,0]`, producing `N-1` evaluations.
/// Timesteps retain the plan-bound training scale for reference consumers.
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
  let evaluation_count = request.model_evaluations()
  if evaluation_count < 1 {
    raise InvalidSchedule
  }
  let shift = scheduler.shift()
  if shift <= 0.0 || shift.is_nan() || shift.is_inf() {
    raise InvalidSchedule
  }
  let point_count = request.inference_steps()
  let denominator = (point_count - 1).to_double()
  let minimum_sigma = 1.0 / scheduler.training_timestep_count().to_double()
  let sigmas : Array[Double] = []
  let timesteps : Array[Double] = []
  for index in 0..<point_count {
    let shifted = match request.sigma_grid() {
      TrainingFloorFloat64 => {
        let fraction = index.to_double() / denominator
        let base = 1.0 + (minimum_sigma - 1.0) * fraction
        shift * base / (1.0 + (shift - 1.0) * base)
      }
      InclusiveZeroFloat32 => {
        let delta : Float = -1.0 / Float::from_int(point_count - 1)
        let base : Float = if index < point_count / 2 {
          Float::from_double(1.0 + delta.to_double() * index.to_double())
        } else {
          Float::from_double(
            0.0 - delta.to_double() * (point_count - index - 1).to_double(),
          )
        }
        let s = Float::from_double(shift)
        let product = (s - 1.0) * base
        let divisor : Float = 1.0 + product
        (s * base / divisor).to_double()
      }
    }
    sigmas.push(shifted)
    if index < evaluation_count {
      timesteps.push(shifted * scheduler.training_timestep_count().to_double())
    }
  }
  if request.sigma_grid() == TrainingFloorFloat64 {
    sigmas.push(0.0)
  }
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
