///|
pub(all) suberror RemoteReleaseError {
  NotReleased
  Binding
  Frame
}

///|
pub struct RemoteReleaseReceipt {
  priv startup : @remote_startup.RemoteRankStartup
}

///|
fn binding(startup : @remote_startup.RemoteRankStartup) -> Bytes raise {
  let bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  ignore(startup.copy_to(bytes))
  Bytes::makei(bytes.length(), i => bytes[i])
}

///|
/// Separate authenticated retirement connection. It grants no execution or
/// lease renewal and is safe to retry after the original connection is lost.
pub async fn receive(
  channel : @remote_tls.AuthenticatedRemoteChannel,
  startup : @remote_startup.RemoteRankStartup,
) -> RemoteReleaseReceipt {
  defer channel.close()
  channel.verify_binding(binding(startup))
  let bytes = FixedArray::makei(16, i => b"LFRELEASEQUERY01"[i])
  channel.send(bytes, 16)
  let length = channel.receive(bytes)
  if length != 16 || Bytes::makei(16, i => bytes[i]) != b"LFRELEASEDONE001" {
    raise Frame
  }
  for i in 0..<16 {
    bytes[i] = b"LFRELEASEACK0001"[i]
  }
  channel.send(bytes, 16)
  { startup, }
}

///|
/// The worker can acknowledge only after its actual deterministic resource
/// owner is closed. Failed cleanup cannot manufacture a release receipt.
pub async fn acknowledge(
  channel : @remote_tls.AuthenticatedRemoteChannel,
  worker : @tensor_parallel_device_worker.TensorParallelDeviceWorkerOwner,
) -> Unit {
  defer channel.close()
  let (startup, _) = worker.remote_contract()
  channel.verify_binding(binding(startup))
  if worker.lifecycle() != WorkerClosed {
    raise NotReleased
  }
  let bytes = FixedArray::make(16, b'\x00')
  let length = channel.receive(bytes)
  if length != 16 || Bytes::makei(16, i => bytes[i]) != b"LFRELEASEQUERY01" {
    raise Frame
  }
  for i in 0..<16 {
    bytes[i] = b"LFRELEASEDONE001"[i]
  }
  channel.send(bytes, 16)
  let length = channel.receive(bytes)
  if length != 16 || Bytes::makei(16, i => bytes[i]) != b"LFRELEASEACK0001" {
    raise Frame
  }
}

///|
pub fn RemoteReleaseReceipt::verify(
  self : RemoteReleaseReceipt,
  expected : @remote_startup.RemoteRankStartup,
) -> Unit raise {
  if binding(self.startup) != binding(expected) {
    raise Binding
  }
}
