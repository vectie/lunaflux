///|
pub(all) enum PagedKvWriteKernelAbi {
  PositionedQkvPagedWriteBf16V1
} derive(Debug, Eq)

///|
pub fn PagedKvWriteKernelAbi::canonical_name(
  self : PagedKvWriteKernelAbi,
) -> String {
  match self {
    PositionedQkvPagedWriteBf16V1 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,positioned_qkv," +
      "key_cache,value_cache,dispatch_canary"
  }
}

///|
pub struct PositionedQkvActivation {
  priv qkv : @plan.ActivationRef
}

///|
pub fn PositionedQkvActivation::new(
  qkv~ : @plan.ActivationRef,
) -> PositionedQkvActivation {
  { qkv, }
}

///|
pub struct PagedKvWriteAotLimits {
  priv max_source_bytes : Int32
  priv max_recipe_bytes : Int32
  priv max_module_bytes : Int64
  priv max_receipt_bytes : Int32
  priv max_binding_bytes : Int32
}

///|
pub fn PagedKvWriteAotLimits::new(
  max_source_bytes~ : Int32,
  max_recipe_bytes~ : Int32,
  max_module_bytes~ : Int64,
  max_receipt_bytes~ : Int32,
  max_binding_bytes~ : Int32,
) -> PagedKvWriteAotLimits raise PagedKvWriteAotError {
  if max_source_bytes <= 0 ||
    max_recipe_bytes <= 0 ||
    max_module_bytes <= 0L ||
    max_receipt_bytes <= 0 ||
    max_binding_bytes <= 0 ||
    max_source_bytes > 1048576 ||
    max_recipe_bytes > 1048576 ||
    max_module_bytes > 1073741824L ||
    max_receipt_bytes > 1048576 ||
    max_binding_bytes > 1048576 {
    raise Invalid(Limits)
  }
  {
    max_source_bytes,
    max_recipe_bytes,
    max_module_bytes,
    max_receipt_bytes,
    max_binding_bytes,
  }
}

///|
pub struct PagedKvWriteCudaCandidate {
  priv model_content : String
  priv model_plan : String
  priv attention_operation : @plan.OperationId
  priv positioned_qkv : @plan.ActivationRef
  priv kv_layer : @plan.KvLayerId
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv tokens_per_page : Int32
  priv total_page_count : Int32
  priv page_stride_bytes : Int64
  priv query_heads : Int32
  priv kv_heads : Int32
  priv head_dimension : Int32
  priv kv_width : Int32
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv symbol : String
  priv abi : PagedKvWriteKernelAbi
  priv dispatch_canary : UInt32
  priv fallback_source : String
  priv fallback_recipe : String
  priv source : Bytes
  priv source_digest : PagedKvWriteDigest
  priv recipe : Bytes
  priv recipe_digest : PagedKvWriteDigest
  priv candidate_digest : PagedKvWriteDigest
}

///|
pub struct PagedKvWriteCompiledArtifact {
  priv candidate : PagedKvWriteDigest
  priv source : PagedKvWriteDigest
  priv recipe : PagedKvWriteDigest
  priv target : @catalog.DeviceTarget
  priv symbol : String
  priv abi : PagedKvWriteKernelAbi
  priv module_bytes : Bytes
  priv module_digest : PagedKvWriteDigest
  priv receipt : Bytes
  priv receipt_digest : PagedKvWriteDigest
  priv driver_identity : String
  priv binding : Bytes
  priv binding_digest : PagedKvWriteDigest
}

///|
/// Source-only qualification export. It carries no compiled artifact,
/// physical observation, manifest entry, promotion, or runtime authority.
pub struct PagedKvWriteQualificationExport {
  priv candidate : PagedKvWriteDigest
  priv source : Bytes
  priv source_digest : PagedKvWriteDigest
  priv recipe : Bytes
  priv recipe_digest : PagedKvWriteDigest
  priv canonical : Bytes
  priv digest : PagedKvWriteDigest
}

///|
/// Caller-reported host/fake campaign observation. It is deliberately not a
/// physical result and cannot grant approval.
pub struct PagedKvWriteFakeCampaignObservation {
  priv live_tokens : Int32
  priv kv_width : Int32
  priv key_writes : Int64
  priv value_writes : Int64
  priv expected_untouched : Int64
  priv observed_untouched : Int64
}

///|
pub fn PagedKvWriteFakeCampaignObservation::new(
  live_tokens~ : Int32,
  kv_width~ : Int32,
  key_writes~ : Int64,
  value_writes~ : Int64,
  expected_untouched~ : Int64,
  observed_untouched~ : Int64,
) -> PagedKvWriteFakeCampaignObservation raise PagedKvWriteAotError {
  if live_tokens <= 0 ||
    kv_width <= 0 ||
    key_writes < 0L ||
    value_writes < 0L ||
    expected_untouched <= 0L ||
    observed_untouched < 0L {
    raise Invalid(Fallback)
  }
  {
    live_tokens,
    kv_width,
    key_writes,
    value_writes,
    expected_untouched,
    observed_untouched,
  }
}

///|
pub struct PagedKvWriteFakeCampaignReport {
  priv candidate : PagedKvWriteDigest
  priv observation_count : Int32
  priv canonical : Bytes
  priv digest : PagedKvWriteDigest
}
