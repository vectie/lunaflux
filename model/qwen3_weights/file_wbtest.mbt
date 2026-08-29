///|
fn qwen3_file_test_metadata() -> @qwen3_spec.Qwen3ModelMetadata {
  try! @qwen3_spec.Qwen3ModelMetadata::from_verified_content(
    @qwen3_spec.Qwen3ModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      head_dimension=4,
      context_length=128,
      rope_theta=1000000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
      tied_embeddings=true,
    ),
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn qwen3_file_append_u64(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn qwen3_file_shape_json(shape : ArrayView[Int64]) -> String {
  let dimensions : Array[String] = []
  for dimension in shape {
    dimensions.push(dimension.to_string())
  }
  "[\{dimensions.join(",")}]"
}

///|
fn qwen3_file_source_fixture(equal_redundant_head? : Bool = true) -> Bytes {
  let metadata = qwen3_file_test_metadata()
  let expected = expected_qwen3_tensors(metadata.spec())
  let header = StringBuilder()
  header.write_string("{")
  let head_bytes = 2L * 8L * 4L
  header.write_string(
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[8,4],\"data_offsets\":[0,\{head_bytes}]}",
  )
  let mut offset = head_bytes
  for tensor in expected {
    let mut length = 2L
    for dimension in tensor.shape {
      length *= dimension
    }
    header.write_string(
      ",\"\{tensor.name}\":{\"dtype\":\"BF16\",\"shape\":\{qwen3_file_shape_json(tensor.shape)},\"data_offsets\":[\{offset},\{offset + length}]}",
    )
    offset += length
  }
  header.write_string("}")
  let header_bytes = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  qwen3_file_append_u64(output, header_bytes.length().to_uint64())
  for byte in header_bytes {
    output.push(byte)
  }
  for index in 0L..<offset {
    let value = if index < head_bytes {
      ((index * 13L + 5L) & 0xffL).to_byte()
    } else if index < head_bytes * 2L {
      let source = index - head_bytes
      ((source * 13L + 5L) & 0xffL).to_byte()
    } else {
      ((index * 17L + 3L) & 0xffL).to_byte()
    }
    output.push(
      if !equal_redundant_head && index == head_bytes {
        (value.to_int() ^ 1).to_byte()
      } else {
        value
      },
    )
  }
  Bytes::from_array(output)
}

///|
fn qwen3_file_source_limits() -> Qwen3SourceFileLimits {
  try! Qwen3SourceFileLimits::new(
    max_file_bytes=131072L,
    compare_chunk_bytes=17,
    safetensors=@safetensors.HeaderLimits::new(
      max_header_bytes=65536,
      max_tensor_count=32,
      max_rank=2,
      max_name_length=128,
      max_payload_bytes=131072L,
      max_materialized_tensor_bytes=131072L,
    ),
  )
}

///|
fn qwen3_file_sha256(file : BytesView) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(file))
}

///|
fn qwen3_numeric_shape_json(shape : @numeric_contract.NumericShape) -> String {
  qwen3_file_shape_json(shape.dimensions())
}

///|
fn qwen3_numeric_fixture(plan : @plan.ModelPlan) -> Bytes {
  let header = StringBuilder()
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan.identity().plan().as_hex()}\",\"numeric_schema_sha256\":\"\{plan.numeric_binding().digest().as_hex()}\"}",
  )
  let mut offset = 0L
  for ordinal, tensor in plan.numeric_binding().schema().tensors() {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    )
    header.write_string(
      ",\"lunaflux.tensor.v1.\{ordinal}\":{\"dtype\":\"BF16\",\"shape\":\{qwen3_numeric_shape_json(tensor.shape())},\"data_offsets\":[\{offset},\{offset + length}]}",
    )
    offset += length
  }
  header.write_string("}")
  let encoded = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  qwen3_file_append_u64(output, encoded.length().to_uint64())
  for byte in encoded {
    output.push(byte)
  }
  for index in 0L..<offset {
    output.push(((index * 19L + 7L) & 0xffL).to_byte())
  }
  Bytes::from_array(output)
}

///|
fn qwen3_numeric_limits() -> @numeric_weight_file.NumericWeightFileLimits {
  try! @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=131072L,
    max_header_bytes=65536,
    transfer_chunk_bytes=257,
    max_tensor_count=32,
  )
}

///|
async test "Qwen3 source inspection proves redundant LM-head byte equality" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-qwen3-source")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let equal = qwen3_file_source_fixture()
    let unequal = qwen3_file_source_fixture(equal_redundant_head=false)
    @fs.write_file(canonical + "/equal", equal, create_mode=CreateNew)
    @fs.write_file(canonical + "/unequal", unequal, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let inspection = inspect_source_safetensors_file(
      root,
      qwen3_file_test_metadata(),
      @approved_fs.ApprovedRelativeLocator::new("equal"),
      expected_artifact_sha256=qwen3_file_sha256(equal),
      limits=qwen3_file_source_limits(),
    )
    assert_eq(inspection.artifact_sha256(), qwen3_file_sha256(equal))
    assert_eq(inspection.source_manifest_sha256().length(), 64)
    assert_true(inspection.bindings().redundant_lm_head_tensor() is Some(_))
    assert_eq(
      inspection.bindings().embedding_binding(),
      inspection.bindings().lm_head_binding(),
    )
    try
      inspect_source_safetensors_file(
        root,
        qwen3_file_test_metadata(),
        @approved_fs.ApprovedRelativeLocator::new("unequal"),
        expected_artifact_sha256=qwen3_file_sha256(unequal),
        limits=qwen3_file_source_limits(),
      )
    catch {
      InvalidTensor(reference=0, TiedOutputByteEquality) => ()
      _ => fail("unexpected redundant-head byte mismatch")
    } noraise {
      _ => fail("unequal redundant LM head was aliased")
    }
    root.close()
  }
}

///|
test "Qwen3 metadata-only binding rejects unproved redundant storage" {
  let file = qwen3_file_source_fixture()
  let tensors = @safetensors_reader.read_metadata(
    file,
    limits=qwen3_file_source_limits().safetensors,
  )
  try bind(qwen3_file_test_metadata(), tensors) catch {
    InvalidTensor(reference=0, TiedOutputByteEquality) => ()
    _ => fail("unexpected metadata-only tied-output rejection")
  } noraise {
    _ => fail("metadata-only binding claimed byte equality")
  }
}

///|
async test "Qwen3 numeric inspection gates exact route manifest and artifact" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-qwen3-numeric")
    let canonical = @fs.realpath(directory)
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(canonical, recursive=true))
    }
    let metadata = qwen3_file_test_metadata()
    let plan = @qwen3.build_paged(metadata)
    let file = qwen3_numeric_fixture(plan)
    @fs.write_file(canonical + "/weights", file, create_mode=CreateNew)
    let batch32 = @qwen3.Qwen3PagedBatchEnvelope::new(32)
    let plan32 = @qwen3.build_paged(metadata, batch=batch32)
    let file32 = qwen3_numeric_fixture(plan32)
    @fs.write_file(canonical + "/weights32", file32, create_mode=CreateNew)
    let artifact = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      qwen3_file_sha256(file),
    )
    let root = @approved_fs.ApprovedRoot::open_absolute(canonical)
    let locator = @approved_fs.ApprovedRelativeLocator::new("weights")
    let generic = @numeric_weight_file.inspect_file(
      root,
      plan,
      locator,
      expected_artifact_digest=artifact,
      limits=qwen3_numeric_limits(),
      alignment_bytes=64L,
      max_arena_bytes=131072L,
    )
    let manifest = qwen3_route_manifest_sha256(plan, generic)
    let inspection = inspect_paged_file(
      root,
      metadata,
      locator,
      expected_artifact_digest=artifact,
      limits=qwen3_numeric_limits(),
      alignment_bytes=64L,
      max_arena_bytes=131072L,
      expected_route_manifest_sha256=manifest,
    )
    assert_eq(inspection.identity(), plan.identity())
    assert_eq(inspection.artifact_digest(), artifact)
    assert_eq(
      inspection.numeric_schema_digest(),
      plan.numeric_binding().digest(),
    )
    assert_eq(inspection.route_manifest_sha256(), manifest)
    let artifact32 = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      qwen3_file_sha256(file32),
    )
    let locator32 = @approved_fs.ApprovedRelativeLocator::new("weights32")
    let generic32 = @numeric_weight_file.inspect_file(
      root,
      plan32,
      locator32,
      expected_artifact_digest=artifact32,
      limits=qwen3_numeric_limits(),
      alignment_bytes=64L,
      max_arena_bytes=131072L,
    )
    let manifest32 = qwen3_route_manifest_sha256(plan32, generic32)
    let inspection32 = inspect_paged_file(
      root,
      metadata,
      locator32,
      expected_artifact_digest=artifact32,
      limits=qwen3_numeric_limits(),
      alignment_bytes=64L,
      max_arena_bytes=131072L,
      expected_route_manifest_sha256=manifest32,
      batch=batch32,
    )
    assert_eq(inspection32.identity(), plan32.identity())
    for
      hostile in [
        (locator, artifact, manifest, batch32),
        (
          locator32,
          artifact32,
          manifest32,
          @qwen3.Qwen3PagedBatchEnvelope::single_row(),
        ),
      ] {
      let (hostile_locator, hostile_artifact, hostile_manifest, hostile_batch) = hostile
      try
        inspect_paged_file(
          root,
          metadata,
          hostile_locator,
          expected_artifact_digest=hostile_artifact,
          limits=qwen3_numeric_limits(),
          alignment_bytes=64L,
          max_arena_bytes=131072L,
          expected_route_manifest_sha256=hostile_manifest,
          batch=hostile_batch,
        )
      catch {
        InvalidNumericWeightFile(_) => ()
        _ => fail("unexpected cross-capacity artifact rejection")
      } noraise {
        _ => fail("numeric artifact crossed its authenticated batch capacity")
      }
    }
    try
      inspect_paged_file(
        root,
        metadata,
        locator,
        expected_artifact_digest=artifact,
        limits=qwen3_numeric_limits(),
        alignment_bytes=64L,
        max_arena_bytes=131072L,
        expected_route_manifest_sha256="0000000000000000000000000000000000000000000000000000000000000000",
      )
    catch {
      Mismatch(RouteManifest) => ()
      _ => fail("unexpected route-manifest mismatch")
    } noraise {
      _ => fail("foreign Qwen3 route manifest was admitted")
    }
    root.close()
  }
}
