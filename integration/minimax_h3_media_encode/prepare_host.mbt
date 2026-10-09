///|
/// Convenience entry for callers that still hold upload staging. Prefer the
/// resolved entry with a WeightManifest captured by model startup.
pub fn AudioRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  artifact : @device.Module,
  source : AudioSource,
  config : @config.MiniMaxH3AudioVaeConfig,
  host : @host.MiniMaxH3HostComponentArena,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity~ : @spec.ModelIdentity,
  input~ : @device.Allocation,
  input_offset~ : Int64,
) -> Unit raise {
  self.prepare_resolved(
    context,
    stream,
    artifact,
    source,
    config,
    WeightManifest::new(host),
    weights,
    expected_identity~,
    input~,
    input_offset~,
  )
}

///|
pub fn VideoRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  artifact : @device.Module,
  source : VideoSource,
  config : @config.MiniMaxH3VideoVaeConfig,
  host : @host.MiniMaxH3HostComponentArena,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity~ : @spec.ModelIdentity,
  input~ : @device.Allocation,
  input_offset~ : Int64,
  noise~ : @device.Allocation,
  noise_offset~ : Int64,
) -> Unit raise {
  self.prepare_resolved(
    context,
    stream,
    artifact,
    source,
    config,
    WeightManifest::new(host),
    weights,
    expected_identity~,
    input~,
    input_offset~,
    noise~,
    noise_offset~,
  )
}
