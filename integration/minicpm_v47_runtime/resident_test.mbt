///|
async test "whole resident owner joins actual checkpoint geometry and closes before startup" {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="lunaflux-v47-resident-headers"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let model = @model.Spec::parse(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
    )
    let operators = @operators.DecoderOperators::new(model, maximum_rows=32)
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
          fail("tensor range")
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
      // Sparse extents support real header inspection; no checkpoint weights
      // are downloaded, materialized or uploaded by this test.
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
    let plan = @weights.Plan::new(
      operators,
      @checkpoint.PackedCheckpoint::new(inspection),
    )
    let resident = @minicpm_v47_runtime.Resident::new(text_program(), plan)
    assert_eq(resident.budget().weight_bytes, plan.byte_count())
    assert_eq(resident.budget().total_allocation_bytes, 69473047868L)
    assert_true(
      try resident.runner() catch {
        _ => true
      } noraise {
        _ => false
      },
    )
    resident.close()
    resident.close()
    assert_true(
      try resident.runner() catch {
        _ => true
      } noraise {
        _ => false
      },
    )
  }
}
