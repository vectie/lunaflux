///|
/// Inspect one component, then stream its packed weights through bounded
/// staging directly to device owners. Text, vision and reference encoders use
/// the same checkpoint slices without ever building full host weight arenas.
pub fn WeightStartup::prepare_component(
  self : Self,
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  sources : ArrayView[@host.MiniMaxH3ShardSource],
  manifest : @host.MiniMaxH3HostComponentManifest,
  packings : ArrayView[Packing],
  expected_identity~ : @spec.ModelIdentity,
  expected_manifest_digest~ : @host.MiniMaxH3HostManifestDigest,
  limits~ : @host.MiniMaxH3HostMaterializationLimits,
  chunk_bytes~ : Int,
) -> Unit raise {
  guard !self.failed && !self.closed && self.leases == 0 && chunk_bytes > 0 else {
    raise InvalidState
  }
  let component = manifest.component()
  guard !self.completed.contains(component) else { raise InvalidState }
  if self.identity is Some(identity) {
    guard identity == expected_identity else { raise InvalidState }
  }
  component_bytes(component, packings, self.maximum - self.used) |> ignore
  self.identity = Some(expected_identity)
  self.failed = true
  self.prepare_streamed_packings(
    context, root, sources, manifest, packings, expected_identity, expected_manifest_digest,
    limits, chunk_bytes,
  )
  self.completed.push(component)
  self.failed = false
}
