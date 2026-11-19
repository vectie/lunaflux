///|
pub extend AudioDecodeRequest with @vectie/lunaflux/engine/joint_diffusion_execution.PreparedFrame::{
  drain,
  release,
}

///|
pub extend ConditioningProgram with @vectie/lunaflux/engine/joint_diffusion_execution.PreparedFrame::{
  drain,
  release,
}

///|
pub extend TextConditioningProgram with @vectie/lunaflux/engine/joint_diffusion_execution.PreparedFrame::{
  drain,
  release,
}
