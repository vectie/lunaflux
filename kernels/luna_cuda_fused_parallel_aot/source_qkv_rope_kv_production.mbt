///|
/// Emits the production QKV/RoPE/KV-write source through a distinct path.
/// Passing no diagnostic canary removes both the ABI argument and publication.
fn fused_qkv_rope_kv_production_source(
  symbol : String,
  parameters : FusedQkvParameters,
  profile : @launch_contract.PagedKernelProfile,
) -> Bytes {
  fused_qkv_rope_kv_source(symbol, parameters, profile, None)
}
