///|
async test "component stream packs tensor slices without host arena padding" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-h3-direct-upload")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let envelope = fixture_envelope(fixture_header(), [1, 2, 3, 4, 5, 6])
    @fs.write_file(canonical + "/weights", envelope, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let manifest = fixture_manifest(envelope)
    let component = inspect_streaming_component(
      root,
      [
        MiniMaxH3ShardSource::new(
          locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
        ),
      ],
      manifest,
      expected_identity=manifest.identity,
      expected_manifest_digest=manifest.digest,
      limits=fixture_limits(),
    )
    let copies = [
      @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name="a",
        source_offset=2L,
        destination_offset=0L,
        byte_count=2L,
      ),
      @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name="a",
        source_offset=0L,
        destination_offset=2L,
        byte_count=2L,
      ),
      @minimax_h3_weights.BlockWeightCopy::new(
        tensor_name="b",
        source_offset=0L,
        destination_offset=4L,
        byte_count=2L,
      ),
    ]
    let requests = component.transfers(copies[:], destination_ordinal=0)
    let packed = FixedArray::make(6, b'!')
    component.transfer(
      root,
      requests[:],
      destination_bytes=[6L],
      scratch=FixedArray::make(1, b'!'),
      consume=(_, offset, scratch, count) => {
        for i in 0..<count {
          packed[offset.to_int() + i] = scratch[i]
        }
        true
      },
    )
    // The original host manifest puts b at offset 64, not offset 4. The stream
    // uses the actual checkpoint tensor and preserves the requested repacking.
    assert_eq(packed, FixedArray::from_array([3, 4, 1, 2, 5, 6]))
    assert_eq(manifest.tensor(1).arena_offset(), 64L)
    root.close()
  }
}
