///|
async fn exercise_checkpoint_index_tables(
  values : ArrayView[Int64],
  invalid~ : Bool,
) -> Unit {
  @async.with_task_group() <| group => {
    let directory = @fs.realpath(@fs.tmpdir(prefix="lunaflux-index-upload"))
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let header = @utf8.encode(
      (
        #|{"first":{"dtype":"I64","shape":[2,3],"data_offsets":[0,48]},"second":{"dtype":"I64","shape":[2,3],"data_offsets":[48,96]}}
      ),
    )
    let file : Array[Byte] = []
    for i in 0..<8 {
      file.push((header.length().to_uint64() >> (8 * i)).to_byte())
    }
    for byte in header {
      file.push(byte)
    }
    for value in values {
      for i in 0..<8 {
        file.push((value.reinterpret_as_uint64() >> (8 * i)).to_byte())
      }
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/weights", bytes, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(directory)
    defer root.close()
    let limits = @streaming.StreamingSafetensorsLimits::new(
      maximum_shards=1,
      maximum_file_bytes=4096L,
      maximum_header_bytes=1024,
      maximum_tensors_per_shard=2,
      maximum_total_tensors=2,
      maximum_rank=2,
      maximum_name_bytes=64,
      maximum_dtype_specs=1,
      maximum_copy_requests=2,
      maximum_copy_bytes=96L,
    )
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new("weights"),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              "a".repeat(64),
            ),
          ),
        ],
        dtype_specs=[
          @streaming.StreamingSafetensorsDTypeSpec::opaque_fixed_width(
            "I64",
            bytes_per_element=8L,
          ),
        ],
        limits~,
      ),
    )
    let tables = [
      checkpoint.bind_i64_table(payload_name="first", rows=2, columns=3),
      checkpoint.bind_i64_table(payload_name="second", rows=2, columns=3),
    ]
    assert_eq(tables[0].input_bytes(), 48L)
    assert_eq(tables[0].i32_bytes(), 24L)
    try
      checkpoint.bind_i64_table(payload_name="first", rows=3, columns=2)
    catch {
      @materialize.PackedCheckpointError::TensorStorageMismatch("first") => ()
      e => raise e
    } noraise {
      _ => fail("transposed source table accepted")
    }
    device_fixture_lane.acquire()
    defer device_fixture_lane.release()
    let baseline = live_resources()
    let context = @device.open_context(0)
    defer (context.close() catch { _ => () })
    for chunk in [1, 3, 9, 17, 64] {
      let owner = @upload.CheckpointDeviceIndexTables::new()
      defer (owner.close() catch { _ => () })
      let host = @upload.i64_table_host_bytes(tables, chunk_bytes=chunk)
      assert_eq(
        host,
        chunk.to_int64() + (chunk.to_int64() + 7L) / 8L * 4L + 48L,
      )
      for (device_budget, host_budget) in [(47L, host), (48L, host - 1L)] {
        let before = live_resources()
        try
          owner.upload(
            context,
            checkpoint,
            root,
            tables,
            minimum_value=0,
            exclusive_maximum_value=8,
            maximum_device_bytes=device_budget,
            maximum_host_bytes=host_budget,
            chunk_bytes=chunk,
            limits~,
          )
        catch {
          @device.DeviceError::InvalidArgument(_) => ()
          e => raise e
        } noraise {
          _ => fail("underbudget index upload accepted")
        }
        assert_eq(owner.device_bytes(), 0L)
        assert_eq(live_resources(), before)
      }
      try
        owner.upload(
          context,
          checkpoint,
          root,
          tables,
          minimum_value=0,
          exclusive_maximum_value=8,
          maximum_device_bytes=48L,
          maximum_host_bytes=host,
          chunk_bytes=chunk,
          limits~,
        )
      catch {
        @narrow.CheckedIntegerNarrowError::DuplicateRowValue(
          row=1,
          first_column=0,
          second_column=2,
          value=3L
        ) if invalid => ()
        @narrow.CheckedIntegerNarrowError::ValueOutOfRange(
          row=1,
          column=2,
          value=8L
        ) if invalid => ()
        e => raise e
      } noraise {
        _ => {
          assert_false(invalid)
          let allocations = owner.allocations()
          assert_eq(allocations.length(), 2)
          for index, allocation in allocations {
            let actual = FixedArray::make(24, b'?')
            context.copy_to_fixed_host(allocation, actual, 0L, 0L, 24L)
            for cell in 0..<6 {
              let expected = values[index * 6 + cell].to_int()
              for byte in 0..<4 {
                assert_eq(
                  actual[cell * 4 + byte],
                  (expected.reinterpret_as_uint() >> (8 * byte)).to_byte(),
                )
              }
            }
          }
        }
      }
      assert_eq(owner.device_bytes(), 48L)
      if invalid {
        try owner.allocations() catch {
          @device.DeviceError::InvalidArgument(_) => ()
          e => raise e
        } noraise {
          _ => fail("failed table upload exposed device storage")
        }
      }
      owner.close()
      owner.close()
      assert_eq(owner.device_bytes(), 0L)
    }
    context.close()
    assert_eq(live_resources(), baseline)
  }
}

///|
async test "checkpoint index tables stream split I64 cells into exact bounded I32 storage" {
  exercise_checkpoint_index_tables(
    [0L, 2L, 1L, 3L, 5L, 4L, 2L, 0L, 1L, 3L, 4L, 5L],
    invalid=false,
  )
}

///|
async test "checkpoint index tables retain failed uploads without publishing malformed rows" {
  exercise_checkpoint_index_tables(
    [0L, 2L, 1L, 3L, 5L, 4L, 2L, 0L, 1L, 3L, 4L, 3L],
    invalid=true,
  )
  exercise_checkpoint_index_tables(
    [0L, 2L, 1L, 3L, 5L, 4L, 2L, 0L, 1L, 3L, 4L, 8L],
    invalid=true,
  )
}
