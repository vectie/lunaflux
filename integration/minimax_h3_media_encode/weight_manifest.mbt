///|
/// Immutable startup metadata, with no host weight arena reference. Capture
/// before releasing upload staging; requests can then bind resident weights
/// without keeping multi-gigabyte checkpoint bytes alive.
pub struct WeightManifest {
  priv component : @model_weights.MiniMaxH3WeightComponent
  priv identity : @spec.ModelIdentity
  priv digest : String
  priv tensors : ReadOnlyArray[@host.MiniMaxH3HostTensor]
}

///|
pub fn WeightManifest::new(
  host : @host.MiniMaxH3HostComponentArena,
) -> Self raise {
  guard !host.is_released() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let tensors : Array[@host.MiniMaxH3HostTensor] = []
  for i in 0..<host.tensor_count() {
    tensors.push(host.tensor(i))
  }
  {
    component: host.component(),
    identity: host.identity(),
    digest: host.manifest_digest().as_hex(),
    tensors: ReadOnlyArray::from_array(tensors),
  }
}
