///|
/// Arithmetic decisions are immutable precision contracts. Only their CUDA
/// spelling belongs here. Matrix logits retain their own separate-round law.
fn control_vector_affine_source(
  index : String,
  scale : String,
  law : @precision.Float32AffineLaw,
) -> String {
  let product = "mixes[" + index + "]"
  let scale = "scale[" + scale + "]"
  let bias = "base[" + index + "]"
  match law {
    SeparateMultiplyAdd =>
      "__fadd_rn(__fmul_rn(" + product + ", " + scale + "), " + bias + ")"
    FusedMultiplyAdd =>
      "__fmaf_rn(" + product + ", " + scale + ", " + bias + ")"
  }
}

///|
fn control_sigmoid_source(law : @precision.SigmoidEvaluationLaw) -> String {
  match law {
    StableBranches =>
      "if (x >= 0.0f) return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); const float e = expf(x); return __fdiv_rn(e, __fadd_rn(1.0f, e));"
    DirectReciprocal => "return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x)));"
  }
}

///|
fn control_root_source(
  argument : String,
  law : @precision.ReciprocalRootLaw,
) -> String {
  match law {
    DivideRoundedSqrtFloat32 => "__fdiv_rn(1.0f, sqrtf(" + argument + "))"
    ReciprocalSqrtFloat32 => "rsqrtf(" + argument + ")"
  }
}
