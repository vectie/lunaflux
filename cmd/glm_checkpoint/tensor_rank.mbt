///|
/// Startup rendezvous on the existing second socket; no residual payload is
/// transferred in expert-TP mode. The generating process stays alive while
/// both communicators bootstrap. All numerical comparison remains external.
async fn connect_tensor_group(
  context : @device.Context,
  stage : @glm.GlmDecoderStage,
  connection : @socket.Tcp,
  rank~ : Int,
) -> @device.TensorCollectiveOwner {
  let runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=21403,
    maximum_version_code=30000,
  )
  let id = @async.with_timeout(300000, () => {
    if rank == 0 {
      let id = runtime.create_group_id()
      let bytes = id.to_bytes()
      for offset = 0; offset < bytes.length(); {
        let count = connection.write_once(
          bytes,
          offset~,
          len=bytes.length() - offset,
        )
        guard count > 0 else { fail("tensor rendezvous disconnected") }
        continue offset + count
      }
      id
    } else {
      // The public backend-neutral group identifier currently encodes 128 bytes.
      let bytes = FixedArray::make(128, b'\x00')
      for offset = 0; offset < bytes.length(); {
        let count = connection.read(
          bytes,
          offset~,
          max_len=bytes.length() - offset,
        )
        guard count > 0 else { fail("tensor rendezvous disconnected") }
        continue offset + count
      }
      @device.tensor_collective_group_id_from_bytes(
        Bytes::from_array(bytes[:])[:],
      )
    }
  })
  // Declared site-table label only, not a computed payload digest.
  let contract = stage.tensor_contract(
    @device.TensorCollectiveContractDigest::from_sha256(
      "0000000000000000000000000000000000000000000000000000000000000001",
    ),
  )
  let owner = context.create_tensor_collective_owner(
    runtime~,
    group_id=id,
    contract~,
    generation=1UL,
    previous_plan_sequence_value=0UL,
  )
  errdefer owner.abort()
  @async.with_timeout(300000, () => {
    while !owner.poll_ready() {
      @async.pause()
    }
  })
  println(
    "expert_tp_ready rank=\{rank} world=2 sites=\{contract.site_count()} reduction=f32-before-round attention=replicated",
  )
  owner
}

///|
async fn run_tensor_egress(
  startup : Startup,
  root : @approved_fs.ApprovedRoot,
  arguments : ArrayView[String],
) -> Unit {
  guard arguments.length() == 4 else {
    fail("expert TP egress currently requires ordinary output mode")
  }
  let context = @device.open_context(0)
  defer close_rank_context(context, "tensor-1")
  let stream = context.create_stream()
  defer stream.close()
  let aot = load_module(context, arguments[0], arguments[1], startup.reserve)
  defer aot.close()
  let stage = stage_plan(startup, 1)
  defer stage.close()
  let rank = @worker.SerialDecoderRank::new(
    new_frame(startup),
    outputs_token=true,
  )
  defer rank.close()
  let listener = @socket.TcpServer::TcpServer(@socket.Addr::parse(arguments[2]))
  defer listener.close()
  let rendezvous = @socket.TcpServer::TcpServer(
    @socket.Addr::parse(arguments[3]),
  )
  defer rendezvous.close()
  println("expert TP waiting for control and rendezvous peers")
  let (connection, _) = listener.accept()
  defer connection.close()
  let (bootstrap, _) = rendezvous.accept()
  defer bootstrap.close()
  let collective = connect_tensor_group(context, stage, bootstrap, rank=1)
  defer collective.abort()
  prepare_rank(startup, 1, context, aot, stream, root, rank, stage, collective~)
  println("expert TP rank=1 loaded; both ranks execute each decoder step")
  let clock = @clock.MonotonicClock::system()
  let server = @control.DecoderStageServer::new(
    rank,
    connection,
    worker_limits(startup),
    inference_limits(startup),
    maximum_buffer_bytes=startup.reserve / 4L,
    timeout_millis=300000UL,
    local_now_millis=clock.now_millis(),
    ready_to_submit=() => true,
  )
  defer server.close()
  while !server.is_terminal() {
    server.poll(local_now_millis=clock.now_millis())
    @async.pause()
  }
}
