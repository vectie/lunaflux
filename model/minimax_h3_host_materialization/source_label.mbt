///|
// Component and arena placement live in the immutable manifest itself.
// Keep the existing plan label without serializing or hashing that metadata.
fn manifest_source_label(
  identity : @spec.ModelIdentity,
) -> MiniMaxH3HostManifestDigest {
  { hex: identity.plan().as_hex(), }
}
