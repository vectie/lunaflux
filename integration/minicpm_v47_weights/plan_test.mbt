///|
async test "all text operands bind actual 16-shard headers without materializing payloads" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-v47-weight-headers"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @model.Spec::parse(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
    )
    let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
    let fixtures = @fs.opendir("model/minicpm_v47_weights/fixtures")
    defer fixtures.close()
    let names = fixtures.read_all()
    names.sort()
    let inputs = []
    for filename in names {
      let header = @fs.read_file(
        "model/minicpm_v47_weights/fixtures/" + filename,
      ).binary()
      guard @json.parse(@utf8.decode(header)) is Object(tree) else {
        fail("header fixture")
      }
      let mut payload_bytes = 0L
      for name, tensor in tree {
        if name == "__metadata__" {
          continue
        }
        guard tensor is { "data_offsets": [_, Number(end, ..)], .. } else {
          fail("tensor offsets")
        }
        payload_bytes = payload_bytes.max(end.to_int64())
      }
      let prefix = @buffer.Buffer()
      prefix.write_uint64_le(header.length().to_uint64())
      prefix.write_bytes(header)
      let path = directory + "/" + filename
      @fs.write_file(path, prefix.to_bytes(), create_mode=CreateNew)
      let file = @fs.open(path, mode=WriteOnly)
      defer file.close()
      // A sparse zero hole supplies the declared extent, never real weights.
      file.write_at(
        b"\x00"[:],
        position=8L + header.length().to_int64() + payload_bytes - 1L,
      )
      inputs.push(
        @streaming.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new(filename),
          expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
            "a".repeat(64),
          ),
        ),
      )
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let inspection = @streaming.inspect_shards(
      root,
      inputs,
      dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::bfloat16()],
      limits=@streaming.StreamingSafetensorsLimits::new(
        maximum_shards=16,
        maximum_file_bytes=80000000000L,
        maximum_header_bytes=1048576,
        maximum_tensors_per_shard=4096,
        maximum_total_tensors=4096,
        maximum_rank=4,
        maximum_name_bytes=256,
        maximum_dtype_specs=1,
        maximum_copy_requests=1024,
        maximum_copy_bytes=2000000000L,
      ),
    )
    assert_eq(inspection.total_tensor_count(), 1152)
    let plan = @minicpm_v47_weights.Plan::new(
      operators,
      @checkpoint.PackedCheckpoint::new(inspection),
    )
    assert_eq(plan.allocations().length(), 653)
    assert_eq(plan.layers().length(), 40)
    assert_true(plan.text().embedding != plan.text().language_head)
    let used : Map[String, Unit] = Map([])
    let mut expected_bytes = 0L
    for allocation in plan.allocations() {
      for name in allocation.sources() {
        assert_true(!used.contains(name))
        assert_true(!name.contains("vision"))
        used[name] = ()
      }
    }
    assert_eq(used.length(), 693)
    for ordinal in 0..<inspection.shard_count() {
      let shard = inspection.shard(ordinal).unwrap()
      for index in 0..<shard.tensor_count() {
        let tensor = shard.tensor(index).unwrap()
        if used.contains(tensor.name()) {
          expected_bytes += tensor.byte_count()
        }
      }
    }
    assert_eq(plan.byte_count(), expected_bytes)
    assert_eq(plan.byte_count(), 69319779584L)
    assert_true(plan.byte_count() < 70425751648L)
    guard plan.layers()[0].branch is Recurrent(recurrent) &&
      plan.layers()[3].branch is FullAttention(full) else {
      fail("branch schedule")
    }
    assert_eq(plan.allocations()[recurrent.convolution].byte_count(), 65536L)
    assert_eq(plan.allocations()[recurrent.qkv].byte_count(), 33554432L)
    assert_eq(plan.allocations()[full.query].byte_count(), 33554432L)
    assert_eq(
      plan.allocations()[plan.layers()[0].moe.routed_bank].byte_count(),
      1610612736L,
    )
    assert_eq(plan.allocations()[plan.text().language_head].sources(), [
      "lm_head.weight",
    ])
    let fresh = @minicpm_v47_weights.Loaded::new(plan)
    fresh.close()
    fresh.close()
    let closed = try fresh.layer_weights() catch {
      _ => true
    } noraise {
      _ => false
    }
    assert_true(closed)
  }
}
