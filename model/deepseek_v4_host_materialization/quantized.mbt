///|
fn quantized_parameter_pair(
  scale : @deepseek_v4_weights.DeepSeekV4TensorSemantic,
  parameter : @deepseek_v4_weights.DeepSeekV4TensorSemantic,
) -> Bool {
  if scale.scope() != parameter.scope() {
    return false
  }
  match (scale.role(), parameter.role()) {
    (Attention(QueryA(QuantizationScale)), Attention(QueryA(QuantizedWeight)))
    | (Attention(QueryB(QuantizationScale)), Attention(QueryB(QuantizedWeight)))
    | (
      Attention(KeyValue(QuantizationScale)),
      Attention(KeyValue(QuantizedWeight)),
    )
    | (
      Attention(OutputA(QuantizationScale)),
      Attention(OutputA(QuantizedWeight)),
    )
    | (
      Attention(OutputB(QuantizationScale)),
      Attention(OutputB(QuantizedWeight)),
    )
    | (
      AttentionIndexer(IndexQuery(QuantizationScale)),
      AttentionIndexer(IndexQuery(QuantizedWeight)),
    )
    | (
      MtpEmbeddingProjection(QuantizationScale),
      MtpEmbeddingProjection(QuantizedWeight),
    )
    | (
      MtpHiddenProjection(QuantizationScale),
      MtpHiddenProjection(QuantizedWeight),
    )
    | (
      DsparkMainProjection(QuantizationScale),
      DsparkMainProjection(QuantizedWeight),
    ) => true
    (
      SharedExpert(scale_matrix, QuantizationScale),
      SharedExpert(parameter_matrix, QuantizedWeight),
    ) => scale_matrix == parameter_matrix
    (
      RoutedExpert(scale_expert, scale_matrix, QuantizationScale),
      RoutedExpert(parameter_expert, parameter_matrix, QuantizedWeight),
    ) => scale_expert == parameter_expert && scale_matrix == parameter_matrix
    _ => false
  }
}

///|
fn logical_parameter_shape(
  tensor : @deepseek_v4_weights.DeepSeekV4SafetensorsTensor,
  encoding : @deepseek_v4_weights.DeepSeekV4WeightEncoding,
) -> Array[Int64] raise DeepSeekV4HostMaterializationError {
  if tensor.shape().length() != 2 {
    raise QuantizedPairUnavailable(-1)
  }
  match encoding {
    Fp8E4M3Block128 => [tensor.shape()[0], tensor.shape()[1]]
    PackedFp4E2M1Block32 => {
      let columns = tensor.shape()[1]
      if columns > 9223372036854775807L / 2L {
        raise QuantizedPairUnavailable(-1)
      }
      [tensor.shape()[0], columns * 2L]
    }
    _ => raise QuantizedPairUnavailable(-1)
  }
}

///|
fn quantized_storage_and_shape(
  weights : DeepSeekV4HostWeights,
  tensor_index : Int,
  tensor : DeepSeekV4HostTensor,
) -> (
  @deepseek_v4_numeric_plan.DeepSeekV4LogicalStorage,
  @deepseek_v4_quantized_payload.DeepSeekV4QuantizedBlockShape,
  Array[Int64],
) raise DeepSeekV4HostMaterializationError {
  let bound = tensor.binding
  match bound.encoding() {
    BFloat16 | Float32 | Int64 => raise TensorNotQuantized(tensor_index)
    Fp8E4M3Block128 =>
      (
        Fp8E4M3Block128Parameter,
        try! @deepseek_v4_quantized_payload.DeepSeekV4QuantizedBlockShape::new(
          rows=128,
          columns=128,
        ),
        logical_parameter_shape(bound.tensor(), bound.encoding()),
      )
    PackedFp4E2M1Block32 =>
      (
        PackedFp4E2M1Block32Parameter,
        try! @deepseek_v4_quantized_payload.DeepSeekV4QuantizedBlockShape::new(
          rows=1,
          columns=32,
        ),
        logical_parameter_shape(bound.tensor(), bound.encoding()),
      )
    Ue8M0Scale => {
      guard tensor.quantized_parameter_index is Some(parameter_index) &&
        weights.tensors.get(parameter_index) is Some(candidate) &&
        quantized_parameter_pair(bound.semantic(), candidate.binding.semantic()) else {
        raise QuantizedPairUnavailable(tensor_index)
      }
      let parameter_encoding = candidate.binding.encoding()
      let shape = logical_parameter_shape(
        candidate.binding.tensor(),
        parameter_encoding,
      )
      match parameter_encoding {
        Fp8E4M3Block128 =>
          (
            Ue8M0Block128ScaleMetadata,
            try! @deepseek_v4_quantized_payload.DeepSeekV4QuantizedBlockShape::new(
              rows=128,
              columns=128,
            ),
            shape,
          )
        PackedFp4E2M1Block32 =>
          (
            Ue8M0Block32ScaleMetadata,
            try! @deepseek_v4_quantized_payload.DeepSeekV4QuantizedBlockShape::new(
              rows=1,
              columns=32,
            ),
            shape,
          )
        _ => raise QuantizedPairUnavailable(tensor_index)
      }
    }
  }
}

///|
pub fn DeepSeekV4HostWeights::quantized_payload_contract(
  self : Self,
  tensor_index : Int,
  limits~ : @deepseek_v4_quantized_payload.DeepSeekV4QuantizedPayloadLimits,
) -> @deepseek_v4_quantized_payload.DeepSeekV4QuantizedPayloadContract raise DeepSeekV4HostMaterializationError {
  let tensor = self.tensor(tensor_index)
  let (storage, block_shape, logical_shape) = quantized_storage_and_shape(
    self, tensor_index, tensor,
  )
  guard self.arenas[tensor.arena_ordinal] is Some(arena) else {
    raise ArenaReleased
  }
  let payload = Bytes::from_array(
    arena[tensor.arena_offset:tensor.arena_offset + tensor.byte_count],
  )
  @deepseek_v4_quantized_payload.admit(
    storage~,
    block_shape~,
    logical_shape~,
    model_content_digest=self.identity.content(),
    payload_bytes=payload,
    limits~,
  ) catch {
    _ => raise QuantizedPayloadFailure(tensor_index)
  }
}

///|
pub fn DeepSeekV4HostWeights::materialization_readiness(
  self : Self,
) -> DeepSeekV4HostMaterializationReadiness {
  let limitations : Array[DeepSeekV4HostExecutionLimitation] = []
  let mut has_fp4 = false
  for tensor in self.tensors {
    if tensor.binding.encoding() is PackedFp4E2M1Block32 {
      has_fp4 = true
    }
  }
  if has_fp4 {
    limitations.push(Fp4ScalarInterpretationUnproven)
  }
  limitations.push(NoDeviceOrKernelMaterialization)
  AuthenticatedRawHostPayloadsOnly(ReadOnlyArray::from_array(limitations))
}

///|
pub fn DeepSeekV4HostWeights::require_executable_weights(
  self : Self,
) -> Unit raise DeepSeekV4HostMaterializationError {
  match self.materialization_readiness() {
    AuthenticatedRawHostPayloadsOnly(limitations) =>
      raise ExecutableWeightsUnavailable(limitations)
  }
}
