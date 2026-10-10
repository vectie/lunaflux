///|
/// One immutable AOT prediction geometry. Source export and device preparation
/// consume this same value; the timestep schedule decides which variants exist.
pub struct PredictionSources {
  block : BlockSource
  timestep : TimestepSource
  output : OutputSource
}

///|
pub fn build_prediction_sources(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  sequence : @minimax_h3_sequence.Sequence,
  distinct_timesteps~ : Int,
  block_norm_epsilon~ : Double,
  function_prefix~ : String,
) -> PredictionSources raise {
  guard distinct_timesteps > 0 && distinct_timesteps <= 4 else {
    raise InvalidShape
  }
  {
    block: build(
      model,
      sequence,
      distinct_timesteps~,
      block_norm_epsilon~,
      function_prefix=function_prefix + "_block_n\{distinct_timesteps}",
    ),
    timestep: build_timestep(
      model,
      distinct_timesteps~,
      function_prefix=function_prefix + "_timestep_n\{distinct_timesteps}",
    ),
    output: build_output(
      model,
      sequence,
      distinct_timesteps~,
      function_prefix=function_prefix + "_output_n\{distinct_timesteps}",
    ),
  }
}
