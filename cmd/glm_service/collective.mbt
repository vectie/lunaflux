///|
async fn connect_collective(
  context : @device.Context,
  stage : @glm.GlmDecoderStage,
  connection : @socket.Tcp,
  ordinal : Int,
) -> @device.TensorCollectiveOwner {
  let runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=21403,
    maximum_version_code=30000,
  )
  let id = @async.with_timeout(300000, () => {
    if ordinal == 0 {
      let id = runtime.create_group_id()
      let bytes = id.to_bytes()
      for offset = 0; offset < bytes.length(); {
        let n = connection.write_once(
          bytes,
          offset~,
          len=bytes.length() - offset,
        )
        guard n > 0 else { fail("TP rendezvous EOF") }
        continue offset + n
      }
      id
    } else {
      let bytes = FixedArray::make(128, b'\x00')
      for offset = 0; offset < 128; {
        let n = connection.read(bytes, offset~, max_len=128 - offset)
        guard n > 0 else { fail("TP rendezvous EOF") }
        continue offset + n
      }
      @device.tensor_collective_group_id_from_bytes(
        Bytes::from_array(bytes[:])[:],
      )
    }
  })
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
  owner
}
