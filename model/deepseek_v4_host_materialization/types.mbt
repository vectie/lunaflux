///|
pub(all) enum DeepSeekV4HostExecutionLimitation {
  Fp4ScalarInterpretationUnproven
  NoDeviceOrKernelMaterialization
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4HostMaterializationReadiness {
  AuthenticatedRawHostPayloadsOnly(
    ReadOnlyArray[DeepSeekV4HostExecutionLimitation]
  )
} derive(Debug, Eq)

///|
pub(all) suberror DeepSeekV4HostMaterializationError {
  InvalidLimit
  IdentityMismatch
  LogicalManifestDigestMismatch
  NumericSchemaDigestMismatch
  NumericBindingDigestMismatch
  ManifestDigestMismatch
  InvalidIndexCount(expected~ : Int32, actual~ : Int32)
  InvalidIndexEntry(Int32)
  DuplicateIndexEntry(Int32, Int32)
  IndexShardMismatch(Int32)
  StreamingFailure
  WeightBindingFailure
  NumericPlanFailure
  TensorTooLarge(Int32)
  HostByteBudgetExceeded
  ArenaCountExceeded
  ArenaReleased
  TensorIndexOutOfBounds(Int32)
  TensorByteOutOfBounds(Int32)
  TensorNotQuantized(Int32)
  QuantizedPairUnavailable(Int32)
  QuantizedPayloadFailure(Int32)
  ExecutableWeightsUnavailable(ReadOnlyArray[DeepSeekV4HostExecutionLimitation])
} derive(Debug, Eq)

///|
pub struct DeepSeekV4HostMaterializationLimits {
  priv maximum_total_host_bytes : Int64
  priv maximum_arena_bytes : Int32
  priv maximum_arenas : Int32
  priv maximum_index_entries : Int32
}

///|
pub struct DeepSeekV4HostExpectedDigests {
  priv identity : @spec.ModelIdentity
  priv logical_manifest_digest : @spec.PlanDigest
  priv numeric_schema_digest : @spec.PlanDigest
  priv numeric_binding_digest : @spec.PlanDigest
} derive(Debug, Eq)

///|
pub struct DeepSeekV4TensorIndexInput {
  priv tensor_name : String
  priv shard_ordinal : Int32
} derive(Debug, Eq)

///|
/// Supplied numeric-binding association label, not a layout or payload checksum.
/// Different arena placements may share this label.
pub struct DeepSeekV4HostManifestDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
priv struct DeepSeekV4HostLayoutTensor {
  binding_index : Int32
  shard_ordinal : Int32
  tensor_ordinal : Int32
  arena_ordinal : Int32
  arena_offset : Int32
  byte_count : Int32
  quantized_parameter_index : Int32?
}

///|
pub struct DeepSeekV4HostManifest {
  priv identity : @spec.ModelIdentity
  priv expected_digests : DeepSeekV4HostExpectedDigests
  priv digest : DeepSeekV4HostManifestDigest
  priv binding : @deepseek_v4_weights.DeepSeekV4WeightBinding
  priv numeric_plan : @deepseek_v4_numeric_plan.DeepSeekV4NumericPlan
  priv inspection : @streaming_safetensors.StreamingSafetensorsInspection
  priv tensors : ReadOnlyArray[DeepSeekV4HostLayoutTensor]
  priv arena_sizes : ReadOnlyArray[Int32]
  priv total_tensor_bytes : Int64
  priv total_arena_bytes : Int64
  priv streaming_limits : @streaming_safetensors.StreamingSafetensorsLimits
}

///|
pub struct DeepSeekV4HostTensor {
  priv binding : @deepseek_v4_weights.DeepSeekV4TensorBinding
  priv arena_ordinal : Int32
  priv arena_offset : Int32
  priv byte_count : Int32
  priv quantized_parameter_index : Int32?
} derive(Debug, Eq)

///|
pub struct DeepSeekV4HostWeights {
  priv identity : @spec.ModelIdentity
  priv expected_digests : DeepSeekV4HostExpectedDigests
  priv manifest_digest : DeepSeekV4HostManifestDigest
  priv tensors : ReadOnlyArray[DeepSeekV4HostTensor]
  priv arenas : Array[FixedArray[Byte]?]
  priv total_tensor_bytes : Int64
  priv total_arena_bytes : Int64
  priv mut released : Bool
}

///|
priv struct HostLayoutSourceTensor {
  shard_ordinal : Int32
  tensor_ordinal : Int32
  byte_count : Int64
}
