///|
/// Small real-driver probe of the production checkpoint transfer/narrowing path.
/// It neither loads a model nor substitutes a C reimplementation of the loader.
async fn main {
  guard @env.args() is [_, directory] else { fail("NEW_OUTPUT_DIRECTORY") }
  @fs.mkdir(directory)
  let header = @utf8.encode(
    (
      #|{"first":{"dtype":"I64","shape":[2,3],"data_offsets":[0,48]},"second":{"dtype":"I64","shape":[2,3],"data_offsets":[48,96]}}
    ),
  )
  let limits = @streaming.StreamingSafetensorsLimits::new(
    maximum_shards=1,
    maximum_file_bytes=4096L,
    maximum_header_bytes=1024,
    maximum_tensors_per_shard=2,
    maximum_total_tensors=2,
    maximum_rank=2,
    maximum_name_bytes=64,
    maximum_dtype_specs=1,
    authentication_chunk_bytes=32,
    maximum_copy_requests=2,
    maximum_copy_bytes=96L,
  )
  let values : Array[Int64] = [0L, 2L, 1L, 3L, 5L, 4L, 2L, 0L, 1L, 3L, 4L, 5L]
  let root = @approved_fs.ApprovedRoot::open_absolute(directory)
  defer root.close()
  let context = @device.open_context(0)
  defer context.close()
  for (name, malformed) in [("valid", false), ("duplicate", true)] {
    let file : Array[Byte] = []
    for byte in 0..<8 {
      file.push((header.length().to_uint64() >> (8 * byte)).to_byte())
    }
    for byte in header {
      file.push(byte)
    }
    for index, value in values {
      let value = if malformed && index == 11 { 3L } else { value }
      for byte in 0..<8 {
        file.push((value.reinterpret_as_uint64() >> (8 * byte)).to_byte())
      }
    }
    let bytes = Bytes::from_array(file)
    @fs.write_file(directory + "/" + name, bytes, create_mode=CreateNew)
    let checkpoint = @materialize.PackedCheckpoint::new(
      @streaming.inspect_shards(
        root,
        [
          @streaming.StreamingSafetensorsShardInput::new(
            locator=@approved_fs.ApprovedRelativeLocator::new(name),
            expected_digest=@streaming.StreamingSafetensorsFileDigest::from_sha256(
              @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
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
    for chunk in [1, 3, 9, 17, 64] {
      let owner = @upload.CheckpointDeviceIndexTables::new()
      defer owner.close()
      let host_bytes = @upload.i64_table_host_bytes(tables, chunk_bytes=chunk)
      for
        (device_budget, host_budget) in [
          (47L, host_bytes),
          (48L, host_bytes - 1L),
        ] {
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
          error => raise error
        } noraise {
          _ => fail("underbudget upload accepted")
        }
        guard owner.device_bytes() == 0L else {
          fail("preflight allocated device memory")
        }
      }
      // Exercise the same pinned-file transfer session used by decoder slices.
      try
        checkpoint.with_transfer_session(root, limits, scoped => {
          owner.upload(
            context,
            scoped,
            root,
            tables,
            minimum_value=0,
            exclusive_maximum_value=8,
            maximum_device_bytes=48L,
            maximum_host_bytes=host_bytes,
            chunk_bytes=chunk,
            limits~,
          )
        })
      catch {
        @narrow.CheckedIntegerNarrowError::DuplicateRowValue(
          row=1,
          first_column=0,
          second_column=2,
          value=3L
        ) if malformed => ()
        error => raise error
      } noraise {
        _ => {
          guard !malformed else { fail("duplicate expert row accepted") }
          for index, allocation in owner.allocations() {
            let output = FixedArray::make(24, b'?')
            context.copy_to_fixed_host(allocation, output, 0L, 0L, 24L)
            for cell in 0..<6 {
              let expected = values[index * 6 + cell].to_int()
              for byte in 0..<4 {
                guard output[cell * 4 + byte] ==
                  (expected.reinterpret_as_uint() >> (8 * byte)).to_byte() else {
                  fail("I32 byte mismatch at table \{index} cell \{cell}")
                }
              }
            }
          }
        }
      }
      guard owner.device_bytes() == 48L else { fail("wrong owned footprint") }
      if malformed {
        try owner.allocations() catch {
          @device.DeviceError::InvalidArgument(_) => ()
          error => raise error
        } noraise {
          _ => fail("failed upload escaped")
        }
      }
      owner.close()
      owner.close()
      guard owner.device_bytes() == 0L else {
        fail("device storage not released")
      }
      println(
        "case=\{name} chunk=\{chunk} source_bytes=96 device_bytes=48 scratch_row_bytes=\{host_bytes} passed",
      )
    }
  }
  context.close()
  println(
    "checkpoint-index-probe-v1 outcome=passed cases=10 exact_i32=true failed_upload_hidden=true released=true",
  )
}
