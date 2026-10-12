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
  @arithmetic.scalar_expression(law.expression(), [product, scale, bias]) catch {
    _ => abort("invalid closed affine program")
  }
}

///|
fn control_sigmoid_source(law : @precision.SigmoidEvaluationLaw) -> String {
  let expression = @arithmetic.scalar_expression(law.expression(), ["x"]) catch {
    _ => abort("invalid closed sigmoid program")
  }
  "return \{expression};"
}

///|
fn control_root_source(
  argument : String,
  law : @precision.ReciprocalRootLaw,
) -> String {
  @arithmetic.scalar_expression(law.expression(), [argument]) catch {
    _ => abort("invalid closed root program")
  }
}
