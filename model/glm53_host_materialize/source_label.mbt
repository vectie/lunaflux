///|
// Retain the already-constructed manifest association. This is not a checksum
// of shard bytes or host segmentation; live regions and bounds remain explicit.
fn host_source_label(
  expected : Glm53HostExpectedDigests,
) -> Glm53HostSourceDigest {
  { hex: expected.manifest_layout.as_hex(), }
}
