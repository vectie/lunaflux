///|
fn float32_role(
  descriptor : @glm53_weights.Glm53TensorDescriptor,
) -> Glm53LogicalTensorRole raise Glm53NumericPlanError {
  let name = descriptor.name()
  if name.contains("e_score_correction_bias") {
    Float32RoutingCorrection
  } else if name.contains(".A_log") || name.contains(".dt_bias") {
    Float32KdaStateParameter
  } else if name.contains(".hc_attn_base") ||
    name.contains(".hc_attn_scale") ||
    name.contains(".hc_ffn_base") ||
    name.contains(".hc_ffn_scale") {
    Float32HyperConnectionParameter
  } else {
    raise UnexpectedFloat32Tensor(ordinal=descriptor.ordinal())
  }
}

///|
fn build_tensor_schemas(
  manifest : @glm53_weights.Glm53TensorManifest,
) -> (
  Array[@numeric_contract.NumericTensorSchema],
  Array[Glm53NumericTensorBinding],
  Int,
  Int,
) raise Glm53NumericPlanError {
  let schemas : Array[@numeric_contract.NumericTensorSchema] = []
  let bindings : Array[Glm53NumericTensorBinding] = []
  let bf16_storage = plain_storage(@numeric_contract.StorageDType::bfloat16())
  let f32_storage = plain_storage(@numeric_contract.StorageDType::f32())
  let mut bf16_count = 0
  let mut f32_count = 0
  for ordinal in 0..<manifest.tensor_count() {
    guard manifest.tensor(ordinal) is Some(descriptor) else {
      raise ManifestOrdinalMismatch(ordinal~)
    }
    if descriptor.ordinal() != ordinal {
      raise ManifestOrdinalMismatch(ordinal~)
    }
    let (role, storage) = match descriptor.dtype() {
      BFloat16 => {
        bf16_count += 1
        (BFloat16Parameter, bf16_storage)
      }
      Float32 => {
        f32_count += 1
        let role = float32_role(descriptor)
        (role, f32_storage)
      }
      Fp8E4M3 =>
        raise UnsupportedFp8BlockScaleLayout(block_rows=128, block_columns=128)
    }
    let schema = @numeric_contract.NumericTensorSchema::new(
      ordinal=@numeric_contract.NumericTensorOrdinal::new(ordinal),
      role=@numeric_contract.NumericTensorRole::parameter(),
      shape=@numeric_contract.NumericShape::new(descriptor.shape()),
      storage~,
    ) catch {
      _ => raise NumericSchemaConstructionFailed
    }
    schemas.push(schema)
    bindings.push({ ordinal, role, })
  }
  (schemas, bindings, bf16_count, f32_count)
}
