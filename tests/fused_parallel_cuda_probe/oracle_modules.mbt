///|
struct FusedProbeOracleModules {
  qkv : Bytes
  rope : Bytes
  kv_write : Bytes
}

///|
fn fused_probe_oracle_modules(
  qkv : Bytes,
  rope : Bytes,
  kv_write : Bytes,
) -> FusedProbeOracleModules raise {
  require_fused_probe(
    !qkv.is_empty() && !rope.is_empty() && !kv_write.is_empty(),
    "standalone oracle CUBIN is empty",
  )
  { qkv, rope, kv_write }
}
