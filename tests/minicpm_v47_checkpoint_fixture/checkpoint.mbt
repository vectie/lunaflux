///|
pub fn limits() -> @streaming.StreamingSafetensorsLimits raise {
  @streaming.StreamingSafetensorsLimits::new(
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
  )
}

///|
/// Actual selected headers with sparse zero payload holes. These files are
/// metadata/byte-transfer fixtures, never checkpoint arithmetic or inference.
pub async fn[T] with_checkpoint(
  body : (@model.Spec, @checkpoint.PackedCheckpoint, @approved_fs.ApprovedRoot) -> T raise,
) -> T {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(
      @fs.tmpdir(prefix="minicpm-v47-header-fixture"),
    )
    group.add_defer(() => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    })
    let spec = @model.Spec::parse(
      @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
    )
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
        fail("fixture header")
      }
      let mut payload = 0L
      for name, tensor in tree {
        if name == "__metadata__" {
          continue
        }
        guard tensor is { "data_offsets": [_, Number(end, ..)], .. } else {
          fail("fixture offsets")
        }
        payload = payload.max(end.to_int64())
      }
      let bytes = @buffer.Buffer()
      bytes.write_uint64_le(header.length().to_uint64())
      bytes.write_bytes(header)
      let path = directory + "/" + filename
      @fs.write_file(path, bytes.to_bytes(), create_mode=CreateNew)
      let file = @fs.open(path, mode=WriteOnly)
      defer file.close()
      file.write_at(
        b"\x00"[:],
        position=8L + header.length().to_int64() + payload - 1L,
      )
      inputs.push(
        @streaming.StreamingSafetensorsShardInput::new(
          locator=@approved_fs.ApprovedRelativeLocator::new(filename),
          expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
            "0".repeat(64),
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
      limits=limits(),
    )
    body(spec, @checkpoint.PackedCheckpoint::new(inspection), root)
  }
}
