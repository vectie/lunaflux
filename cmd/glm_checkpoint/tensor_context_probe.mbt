///|
/// Small real-ABI diagnostic. No model weights or reference implementation is
/// loaded. Tests both context policies against the identical communicator.
async fn probe_tensor_context(
  rank : Int,
  endpoint : String,
  interoperable : Bool,
) -> Unit {
  guard rank is (0 | 1) else { fail("rank must be zero or one") }
  let context = @device.open_context(0, runtime_interoperable=interoperable)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let buffer = context.allocate(128L)
  defer buffer.close()
  let bits = Float::from_int(rank + 1).reinterpret_as_uint()
  let input = FixedArray::makei(128, i => {
    ((bits >> (i % 4 * 8)) & 255U).to_byte()
  })
  context.copy_from_fixed_host(buffer, input, 0L, 0L, 128L)
  let listener = if rank == 1 {
    Some(@socket.TcpServer::TcpServer(@socket.Addr::parse(endpoint)))
  } else {
    None
  }
  defer (if listener is Some(server) { server.close() })
  let connection = match listener {
    Some(server) => server.accept().0
    None => @socket.Tcp::connect(@socket.Addr::parse(endpoint))
  }
  defer connection.close()
  let contract = @device.TensorCollectiveContract::new(
    digest=@device.TensorCollectiveContractDigest::from_sha256(
      "0000000000000000000000000000000000000000000000000000000000000001",
    ),
    rank~,
    world_size=2,
    max_query_tokens=2,
    sites=[
      @device.TensorCollectiveSite::new(
        sequence=1,
        operation_id=0,
        kind=F32SumAllReduce,
        full_width=16,
        local_width=16,
      ),
    ],
  )
  let owner = connect_tensor_contract(context, contract, connection, rank~)
  defer owner.abort()
  for sequence in 1..<5 {
    context.copy_from_fixed_host(buffer, input, 0L, 0L, 128L)
    owner.submit(
      generation=1UL,
      plan_sequence=sequence.to_uint64(),
      global_sequence=sequence.to_uint64(),
      site_sequence=1,
      operation_id=0,
      kind=F32SumAllReduce,
      live_query_tokens=2,
      send=buffer,
      send_offset=0L,
      receive=buffer,
      receive_offset=0L,
      stream~,
    )
    @async.with_timeout(30000, () => {
      while !owner.poll_submitted() {
        @async.pause()
      }
      while !owner.poll() {
        @async.pause()
      }
    })
    let output = FixedArray::make(128, b'\x00')
    context.copy_to_fixed_host(buffer, output, 0L, 0L, 128L)
    let expected = 3.0F.reinterpret_as_uint()
    for i in 0..<128 {
      assert_eq(output[i], ((expected >> (i % 4 * 8)) & 255U).to_byte())
    }
  }
  owner.close()
  println(
    "tensor_context_probe rank=\{rank} runtime_interoperable=\{interoperable} all32values=exact repetitions=4",
  )
}
