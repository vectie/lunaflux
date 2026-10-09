///|
/// Composes exact rank-specific envelopes from root-free local worker contracts.
/// Callers authenticate deployment inputs separately. No node is contacted,
/// no root or CUDA handle is acquired, and no Ready state is produced.
pub fn RemoteGroupStartup::new(
  plan : @two_node_plan.TwoNodePlan,
  workers : ArrayView[@worker_wire.WorkerStartupContract],
  executables : ArrayView[@spec.ContentDigest],
  deployment_digest~ : @spec.ContentDigest,
  group_generation~ : UInt64,
  rendezvous~ : @device.TensorCollectiveGroupId,
  lease_millis~ : UInt64,
) -> RemoteGroupStartup raise RemoteStartupError {
  if workers.length() != 2 || executables.length() != 2 {
    raise Rejected(Count)
  }
  if group_generation == 0UL {
    raise Rejected(Generation)
  }
  if lease_millis == 0UL || lease_millis > 300000UL {
    raise Rejected(Lease)
  }
  if rendezvous.to_bytes().length() != 128 {
    raise Rejected(Rendezvous)
  }
  for rank, worker in workers {
    guard plan.topology().rank(rank) is Some(node) else {
      raise Rejected(Count)
    }
    if worker.identity() != plan.model_identity() {
      raise Rejected(Model)
    }
    if worker.device_ordinal() != node.device_ordinal() {
      raise Rejected(Ordinal)
    }
    if worker.model_generation_value() != workers[0].model_generation_value() {
      raise Rejected(Generation)
    }
    if worker.has_predecessor() != workers[0].has_predecessor() ||
      worker.predecessor_value() != workers[0].predecessor_value() {
      raise Rejected(Predecessor)
    }
    if worker.worker_limits() != workers[0].worker_limits() ||
      worker.inference_limits() != workers[0].inference_limits() {
      raise Rejected(Limits)
    }
  }
  let frames : Array[FixedArray[Byte]] = []
  for rank in 0..<2 {
    frames.push(
      encode_rank(
        plan,
        workers[rank],
        executables[rank],
        deployment_digest,
        group_generation,
        rendezvous,
        lease_millis,
        rank,
      ),
    )
  }
  // Bind both local artifact/executable contracts into one generation identity.
  // The initially zero group-digest field avoids recursive self-hashing.
  let hash = @canonical_sha256.CanonicalSha256::new()
  ignore(hash.update_bytes(b"lunaflux.remote-group-startup.v1"))
  for frame in frames {
    for index in 0..<CHECKSUM_OFFSET {
      ignore(hash.update_byte(frame[index]))
    }
  }
  let digest = hash.finalize_hex()
  let ranks : Array[RemoteRankStartup] = []
  for rank, frame in frames {
    put_text(frame, GROUP_DIGEST_OFFSET, digest)
    seal(frame)
    guard plan.topology().rank(rank) is Some(node) else {
      raise Rejected(Count)
    }
    ranks.push({
      plan_digest: plan.digest(),
      frame: Bytes::from_array(frame),
      worker: workers[rank],
      node,
      rank,
      group_generation,
      group_digest: digest,
      lease_millis,
      rendezvous,
    })
  }
  { ranks: ReadOnlyArray::from_array(ranks), digest, }
}
