///|
pub fn streaming_limits() -> @streaming.StreamingSafetensorsLimits raise {
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
/// Read startup assets under a size ceiling, with no payload integrity pass.
pub fn read_asset(
  root : @approved_fs.ApprovedRoot,
  name : String,
  maximum_bytes~ : Int64,
) -> Bytes raise {
  let file = root.open_file(@approved_fs.ApprovedRelativeLocator::new(name))
  defer file.close()
  file.read_immutable_snapshot(maximum_bytes~)
}

///|
/// Inspect bounded shard headers and join the index's ownership map. No tensor
/// payload is read and the supplied association label is not a payload hash.
pub fn inspect_checkpoint(
  root : @approved_fs.ApprovedRoot,
  document : Json,
) -> @streaming.StreamingSafetensorsInspection raise {
  guard document is { "weight_map": Object(index), .. } &&
    index.length() > 0 &&
    index.length() <= 4096 else {
    fail("invalid selected-model tensor index")
  }
  let unique : Map[String, Unit] = Map([])
  for _, owner in index {
    guard owner is String(name) && name.has_suffix(".safetensors") else {
      fail("invalid shard owner")
    }
    unique[name] = ()
  }
  let names = unique.keys().to_array()
  names.sort()
  guard names.length() > 0 && names.length() <= 16 else {
    fail("invalid selected-model shard count")
  }
  let label = @streaming.StreamingSafetensorsFileDigest::from_sha256(
    "0".repeat(64),
  )
  let inputs = names.map(name => {
    @streaming.StreamingSafetensorsShardInput::new(
      locator=@approved_fs.ApprovedRelativeLocator::new(name),
      expected_digest=label,
    )
  })
  let inspection = @streaming.inspect_shards(
    root,
    inputs,
    dtype_specs=[@streaming.StreamingSafetensorsDTypeSpec::bfloat16()],
    limits=streaming_limits(),
  )
  guard inspection.total_tensor_count() == index.length() else {
    fail("tensor index/header count mismatch")
  }
  for shard in 0..<inspection.shard_count() {
    let metadata = inspection.shard(shard).unwrap()
    for tensor in 0..<metadata.tensor_count() {
      guard index.get(metadata.tensor(tensor).unwrap().name()) ==
        Some(Json(names[shard])) else {
        fail("tensor index/header shard mismatch")
      }
    }
  }
  inspection
}
