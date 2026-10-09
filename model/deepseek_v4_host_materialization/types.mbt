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
  InvalidIndexCount(expected~ : Int, actual~ : Int)
  InvalidIndexEntry(Int)
  DuplicateIndexEntry(Int, Int)
  IndexShardMismatch(Int)
  StreamingFailure
  WeightBindingFailure
  NumericPlanFailure
  TensorTooLarge(Int)
  HostByteBudgetExceeded
  ArenaCountExceeded
  ArenaReleased
  TensorIndexOutOfBounds(Int)
  TensorByteOutOfBounds(Int)
  TensorNotQuantized(Int)
  QuantizedPairUnavailable(Int)
  QuantizedPayloadFailure(Int)
  ExecutableWeightsUnavailable(ReadOnlyArray[DeepSeekV4HostExecutionLimitation])
} derive(Debug, Eq)

///|
pub struct DeepSeekV4HostMaterializationLimits {
  priv maximum_total_host_bytes : Int64
  priv maximum_arena_bytes : Int
  priv maximum_arenas : Int
  priv maximum_index_entries : Int
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
  priv shard_ordinal : Int
} derive(Debug, Eq)

///|
pub struct DeepSeekV4HostManifestDigest {
  priv hex : String
} derive(Debug, Eq, Hash)

///|
priv struct DeepSeekV4HostLayoutTensor {
  binding_index : Int
  shard_ordinal : Int
  tensor_ordinal : Int
  arena_ordinal : Int
  arena_offset : Int
  byte_count : Int
  quantized_parameter_index : Int?
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
  priv arena_sizes : ReadOnlyArray[Int]
  priv total_tensor_bytes : Int64
  priv total_arena_bytes : Int64
  priv streaming_limits : @streaming_safetensors.StreamingSafetensorsLimits
}

///|
pub struct DeepSeekV4HostTensor {
  priv binding : @deepseek_v4_weights.DeepSeekV4TensorBinding
  priv arena_ordinal : Int
  priv arena_offset : Int
  priv byte_count : Int
  priv quantized_parameter_index : Int?
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
  shard_ordinal : Int
  tensor_ordinal : Int
  byte_count : Int64
}
