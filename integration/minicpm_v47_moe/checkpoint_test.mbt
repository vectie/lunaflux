///|
async test "real sliced streaming transfer preserves packed expert axes with three-byte scratch" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-minicpm-experts")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let header_text =
      #|{"model.language_model.layers.0.mlp.experts.gate_up_proj":{"dtype":"BF16","shape":[2,4,2],"data_offsets":[0,32]},"model.language_model.layers.0.mlp.experts.down_proj":{"dtype":"BF16","shape":[2,2,2],"data_offsets":[32,48]}}
    let header = @utf8.encode(header_text)
    let payload = FixedArray::makei(48, i => i.to_byte())
    let file = @buffer.Buffer()
    file.write_uint64_le(header.length().to_uint64())
    file.write_bytes(header)
    file.write_bytes(Bytes::from_array(payload))
    let bytes = file.to_bytes()
    @fs.write_file(canonical + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=65536L,
      maximum_header_bytes=4096,
      maximum_tensors_per_shard=2,
      maximum_total_tensors=2,
      maximum_rank=3,
      maximum_name_bytes=128,
      maximum_dtype_specs=1,
      maximum_copy_requests=32,
      maximum_copy_bytes=1024L,
    )
    let inspection = @streaming.inspect_shards(
      root,
      [
        @streaming.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
          expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
            @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
          ),
        ),
      ],
      dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::bfloat16()],
      limits~,
    )
    let checkpoint = @checkpoint.PackedCheckpoint::new(inspection)
    let operand = @precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(format=Bf16, rows=2, columns=2),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision.Accuracy::new(absolute=0.0, relative=0.0),
      memory_ceiling=100000L,
      tile_elements=32,
    )
    let plan = @precision.ExpertMlpPrecision::new(
      gate=operand,
      up=operand,
      down=operand,
      expert_count=2,
      local_experts=[0, 1],
      selected_experts=2,
      maximum_rows=1,
      combine_rounding=Bf16EachAdd,
      memory_ceiling=100000L,
    )
    let slices = @minicpm_v47_moe.bind_expert_weights(checkpoint, plan, layer=0)
    assert_eq(slices.length(), 6)
    assert_eq(plan.bank_bytes(), 96L)
    let bank = FixedArray::make(96, b'\x00')
    let destinations = slices.mapi((i, _) => {
      (
        0,
        (i / 3).to_int64() * plan.expert_stride() + plan.weight_offsets()[i % 3],
      )
    })
    let mut chunks = 0
    checkpoint.transfer_weights_to(
      root,
      slices[:],
      destinations~,
      destination_bytes=[plan.bank_bytes()],
      scratch=FixedArray::make(3, b'\x00'),
      limits~,
      consume=(_, offset, source, count) => {
        guard count <= 3 else { return false }
        for i in 0..<count {
          bank[offset.to_int() + i] = source[i]
        }
        chunks += 1
        true
      },
    )
    let expected = FixedArray::make(96, b'\x00')
    for
      (destination, source) in [
        (0, 0),
        (16, 8),
        (32, 32),
        (48, 16),
        (64, 24),
        (80, 40),
      ] {
      for i in 0..<8 {
        expected[destination + i] = payload[source + i]
      }
    }
    assert_eq(Bytes::from_array(bank), Bytes::from_array(expected))
    assert_true(chunks > 6)
  }
}
