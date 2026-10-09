///|
/// Runs one operator-launched rank from a digest-pinned local deployment root.
/// After execution ends, retains a release-only listener for recovery receipts.
/// It cannot execute another generation; deployment starts a fresh process.
pub async fn run(
  deployment_root : String,
  expected_sha256 : String,
  parallel_policy? : @parallel_ir.Policy = Ordered,
) -> Unit {
  let root = @approved_fs.ApprovedRoot::open_absolute(deployment_root)
  defer (root.close() catch { _ => () })
  let config = document(
    snapshot(root, "two-spark.json", 65536L),
    expected_sha256,
  )
  let source_bytes = pinned(root, field(config, "source"), 1048576L)
  let source = @worker_wire.decode_bootstrap_source(
    fixed(source_bytes),
    source_bytes.length(),
  )
  let workers = pair(config, "workers").map(reference => {
    worker_contract(root, reference)
  })
  let rank_value = number(config, "rank")
  if rank_value > 1L {
    raise Configuration
  }
  let rank = rank_value.to_int()
  let envelope_bytes = pinned(root, field(config, "envelope"), 65536L)
  let envelope = @bootstrap.TensorParallelRankEnvelopeFrameBuffer::new()
    .load(fixed(envelope_bytes), envelope_bytes.length())
    .envelope()
  if envelope.rank() != rank {
    raise Configuration
  }
  let execution = source.execution()
  let target = @catalog.DeviceTarget::new(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  )
  let nodes = pair(config, "nodes").map(value => network_node(value, target))
  let topology = @network_topology.TwoNodeTopology::new(nodes[0], nodes[1])
  let budgets = pair(config, "budgets").map(memory)
  let model_root = @approved_fs.ApprovedRoot::open_absolute(
    text(config, "model_root"),
  )
  defer (model_root.close() catch { _ => () })
  let kernel_root = @approved_fs.ApprovedRoot::open_absolute(
    text(config, "kernel_root"),
  )
  defer (kernel_root.close() catch { _ => () })
  let compiled = @child.compile_remote_plan(
    model_root,
    source,
    workers[rank],
    topology,
    budgets,
  )
  let executables = pair(config, "executables").map(value => {
    guard value is String(digest) else { raise Configuration }
    @spec.ContentDigest::from_sha256(digest)
  })
  let lease = number(config, "lease_millis")
  if lease < 30L || lease > 300000L {
    raise Configuration
  }
  let remote = @remote_startup.RemoteGroupStartup::new(
    compiled,
    workers,
    executables,
    deployment_digest=@spec.ContentDigest::from_sha256(
      text(config, "deployment_sha256"),
    ),
    group_generation=envelope.group_generation_value(),
    rendezvous=envelope.rendezvous_id(),
    lease_millis=lease.reinterpret_as_uint64(),
  )
  let startup = remote.rank(rank).unwrap()
  let plan = @child.load_remote_rank_plan(
    model_root,
    kernel_root,
    source,
    compiled,
    startup,
    envelope,
    envelope.group_digest(),
  )
  let runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=envelope.collective_runtime_version_code(),
    maximum_version_code=envelope.collective_runtime_version_code(),
  )
  let maximum = number(config, "max_payload_bytes")
  if maximum < 1112L || maximum > 67100000L {
    raise Configuration
  }
  let limits = @rank_group_wire.RankGroupWireLimits::new(
    max_payload_bytes=maximum.to_int(),
  )
  let credential = @remote_tls.RemoteCredential::from_bytes(
    pinned(root, field(config, "credential"), 32L),
  )
  ignore(pinned(root, field(config, "certificate"), 1048576L))
  ignore(pinned(root, field(config, "private_key"), 1048576L))
  guard field(config, "certificate") is Object(certificate) else {
    raise Configuration
  }
  guard field(config, "private_key") is Object(private_key) else {
    raise Configuration
  }
  let certificate_path = deployment_root + "/" + text(certificate, "path")
  let private_key_path = deployment_root + "/" + text(private_key, "path")
  let binding_bytes = FixedArray::make(
    @remote_startup.remote_startup_frame_bytes(),
    b'\x00',
  )
  ignore(startup.copy_to(binding_bytes))
  let binding = Bytes::makei(binding_bytes.length(), i => binding_bytes[i])
  let listener = @socket.TcpServer::TcpServer(
    @socket.Addr::parse(text(config, "listen")),
  )
  defer listener.close()
  let worker = match
    @worker.prepare(model_root, runtime, plan, parallel_policy~) {
    TensorParallelWorkerPrepared(worker) => worker
    TensorParallelWorkerCleanupRequired(failed) => {
      failed.retry_close()
      raise Cleanup
    }
  }
  defer (worker.close() catch { _ => () })
  try {
    let channel = @async.with_timeout(lease.to_int(), () => {
      let (socket, _) = listener.accept()
      @remote_tls.accept(
        socket,
        credential,
        binding,
        certificate_pem_file=certificate_path,
        private_key_pem_file=private_key_path,
        max_frame_bytes=limits.max_frame_bytes() + 1,
        timeout_millis=lease.to_int(),
      )
    })
    @remote_rank_worker.serve(channel, worker, limits)
  } catch {
    _ => ()
  }
  // Explicit retry authority stays with this process until release succeeds.
  worker.close()
  if worker.lifecycle() != WorkerClosed {
    raise Cleanup
  }
  for ;; {
    let (socket, _) = listener.accept()
    let channel = @remote_tls.accept(
      socket,
      credential,
      binding,
      certificate_pem_file=certificate_path,
      private_key_pem_file=private_key_path,
      max_frame_bytes=limits.max_frame_bytes() + 1,
      timeout_millis=lease.to_int(),
    ) catch {
      _ => continue
    }
    @remote_release.acknowledge(channel, worker) catch {
      _ => ()
    }
  }
}
