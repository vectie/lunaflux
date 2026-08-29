///|
suberror KvWriteProbeRejected {
  KvWriteProbeRejected(String)
} derive(Debug, Eq)

///|
fn require_kv_probe(
  condition : Bool,
  reason : String,
) -> Unit raise KvWriteProbeRejected {
  if !condition {
    raise KvWriteProbeRejected(reason)
  }
}
