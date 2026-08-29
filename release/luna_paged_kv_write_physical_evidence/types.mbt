///|
const PAGED_KV_WRITE_PHYSICAL_MAX_BYTES : Int = 131072

///|
priv struct PagedKvWritePhysicalReader {
  lines : Array[String]
  mut cursor : Int
}

///|
pub struct QualifiedPagedKvWriteArtifact {
  priv candidate : @luna_cuda_paged_kv_write_aot.PagedKvWriteCudaCandidate
  priv compiled : @luna_cuda_paged_kv_write_aot.PagedKvWriteCompiledArtifact
  priv evidence_digest : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest
  priv approved_policy_digest : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest
  priv device_uuid : String
  priv device_pci : String
  priv files_manifest_digest : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest
  priv outer_seal_digest : @luna_cuda_paged_kv_write_aot.PagedKvWriteDigest
}
