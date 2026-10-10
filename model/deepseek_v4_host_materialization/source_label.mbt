///|
// Association metadata only: the immutable manifest owns its actual layout.
// A source label must not serialize or authenticate that layout again.
fn manifest_source_label(
  expected : DeepSeekV4HostExpectedDigests,
) -> DeepSeekV4HostManifestDigest {
  { hex: expected.numeric_binding_digest.as_hex(), }
}
