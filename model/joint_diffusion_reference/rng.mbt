///|
fn digest_hash(digest : @spec.PlanDigest) -> UInt64 {
  let text = digest.as_hex()
  let mut hash = 14695981039346656037UL
  for index in 0..<text.length() {
    hash = (hash ^ text[index].to_uint64()) * 1099511628211UL
  }
  hash
}

///|
fn splitmix64(input : UInt64) -> UInt64 {
  let mut value = input + 0x9e3779b97f4a7c15UL
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL
  value ^ (value >> 31)
}

///|
fn stream_domain(stream : JointLatentRandomStream) -> UInt64 {
  match stream {
    ConditioningStream => 0x243f6a8885a308d3UL
    VideoLatentStream => 0x13198a2e03707344UL
    AudioLatentStream => 0xa4093822299f31d0UL
  }
}

///|
fn uniform_open(value : UInt64) -> Double {
  ((value >> 11).to_double() + 0.5) / 9007199254740992.0
}

///|
/// Constructs three disjoint, stateless random streams. The output for one
/// `(stream, counter)` pair is independent of call order and does not mutate
/// shared RNG state.
pub fn CounterBasedJointLatentRng::new(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  request : @joint_diffusion_plan.JointDiffusionRequestShape,
  seed~ : Int64,
  conditioning_draw_limit~ : Int64,
) -> CounterBasedJointLatentRng raise JointDiffusionReferenceError {
  if plan.schema_version() != 1 ||
    !plan.conditioning_draws_precede_output_draws() ||
    !request_matches_plan(plan, request) {
    raise InvalidPlanContract
  }
  if conditioning_draw_limit < 0L || conditioning_draw_limit > 1000000000L {
    raise InvalidConditioningDrawLimit
  }
  let shapes = plan.shapes()
  let video_draw_limit = checked_product([
    request.video_latent_frames().to_int64(),
    request.video_latent_height().to_int64(),
    request.video_latent_width().to_int64(),
    shapes.video_latent_channels().to_int64(),
  ])
  let audio_draw_limit = checked_product([
    request.audio_token_rows().to_int64(),
    shapes.audio_latent_channels().to_int64(),
  ])
  let plan_digest = plan.plan_digest()
  let key = seed.reinterpret_as_uint64() ^ digest_hash(plan_digest)
  {
    plan_digest,
    key,
    conditioning_draw_limit,
    video_draw_limit,
    audio_draw_limit,
  }
}

///|
fn CounterBasedJointLatentRng::limit(
  self : Self,
  stream : JointLatentRandomStream,
) -> Int64 {
  match stream {
    ConditioningStream => self.conditioning_draw_limit
    VideoLatentStream => self.video_draw_limit
    AudioLatentStream => self.audio_draw_limit
  }
}

///|
/// Returns one deterministic standard-normal sample using counter-keyed
/// SplitMix64 lanes and a Box-Muller transform.
pub fn CounterBasedJointLatentRng::normal(
  self : Self,
  stream : JointLatentRandomStream,
  counter : Int64,
) -> Double raise JointDiffusionReferenceError {
  if counter < 0L || counter >= self.limit(stream) {
    raise InvalidRandomCounter(stream)
  }
  let counter_bits = counter.reinterpret_as_uint64()
  let base = self.key ^
    stream_domain(stream) ^
    (counter_bits * 0x9e3779b97f4a7c15UL)
  let first = uniform_open(splitmix64(base ^ 0xd1b54a32d192ed03UL))
  let second = uniform_open(splitmix64(base ^ 0x94d049bb133111ebUL))
  (-2.0 * @math.ln(first)).sqrt() * @math.cos(2.0 * @math.PI * second)
}

///|
pub fn CounterBasedJointLatentRng::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn CounterBasedJointLatentRng::draw_limit(
  self : Self,
  stream : JointLatentRandomStream,
) -> Int64 {
  self.limit(stream)
}
