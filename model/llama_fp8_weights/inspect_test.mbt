///|
fn fp8_file_metadata() -> @spec.LlamaModelMetadata {
  try! @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=16,
      hidden_size=4,
      intermediate_size=8,
      layer_count=2,
      attention_head_count=2,
      kv_head_count=1,
      context_length=16,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0",
    ),
  )
}

///|
fn fp8_file_plan() -> @plan.ModelPlan {
  try! @llama.build_finite_e4m3_fp8_w8a8_v1(
    fp8_file_metadata(),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
  )
}

///|
fn append_u64_le(output : Array[Byte], value : UInt64) -> Unit {
  for shift in 0..<8 {
    output.push(((value >> (shift * 8)) & 0xffUL).to_byte())
  }
}

///|
fn shape_json(
  tensor : @numeric_contract.NumericTensorSchema,
  wrong : Bool,
) -> String {
  let dimensions : Array[String] = []
  for axis, dimension in tensor.shape().dimensions() {
    dimensions.push(
      if wrong && axis == 0 {
        (dimension + 1L).to_string()
      } else {
        dimension.to_string()
      },
    )
  }
  "[\{dimensions.join(",")}]"
}

///|
fn dtype_name(
  tensor : @numeric_contract.NumericTensorSchema,
  fp8_alias : Bool,
) -> String {
  let dtype = tensor.storage().dtype()
  if dtype == @numeric_contract.StorageDType::bfloat16() {
    "BF16"
  } else if dtype == @numeric_contract.StorageDType::f8_e4m3_finite() {
    if fp8_alias {
      "F8E4M3"
    } else {
      "F8_E4M3"
    }
  } else {
    "F32"
  }
}

///|
fn fp8_numeric_file(
  plan : @plan.ModelPlan,
  alias_name_ordinal? : Int = -1,
  fp8_alias_ordinal? : Int = -1,
  omit_ordinal? : Int = -1,
  overlap_ordinal? : Int = -1,
  wrong_shape_ordinal? : Int = -1,
  extra? : Bool = false,
  wrong_schema? : Bool = false,
) -> Bytes {
  let tensors = plan.numeric_binding().schema().tensors()
  let schema_digest = if wrong_schema {
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
  } else {
    plan.numeric_binding().digest().as_hex()
  }
  let header = StringBuilder()
  header.write_string(
    "{\"__metadata__\":{\"format\":\"lunaflux.numeric-safetensors.v1\",\"model_content_sha256\":\"\{plan.identity().content().as_hex()}\",\"model_plan_sha256\":\"\{plan.identity().plan().as_hex()}\",\"numeric_schema_sha256\":\"\{schema_digest}\"}",
  )
  let mut offset = 0L
  for ordinal, tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    )
    if ordinal != omit_ordinal {
      let name = if ordinal == alias_name_ordinal {
        "lunaflux.tensor.v1.0\{ordinal}"
      } else {
        "lunaflux.tensor.v1.\{ordinal}"
      }
      let start = if ordinal == overlap_ordinal { offset - 1L } else { offset }
      header.write_string(
        ",\"\{name}\":{\"dtype\":\"\{dtype_name(tensor, ordinal == fp8_alias_ordinal)}\",\"shape\":\{shape_json(tensor, ordinal == wrong_shape_ordinal)},\"data_offsets\":[\{start},\{start + length}]}",
      )
    }
    offset += length
  }
  if extra {
    header.write_string(
      ",\"unexpected\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[\{offset},\{offset + 4L}]}",
    )
  }
  header.write_string("}")
  let encoded_header = @utf8.encode(header.to_string())
  let output : Array[Byte] = []
  append_u64_le(output, encoded_header.length().to_uint64())
  for byte in encoded_header {
    output.push(byte)
  }
  for tensor in tensors {
    let length = try! @numeric_contract.storage_byte_length(
      tensor.storage().dtype(),
      tensor.shape(),
    ).to_int()
    if tensor.storage().dtype() == @numeric_contract.StorageDType::f32() {
      for _ in 0..<(length / 4) {
        output.push(b'\x00')
        output.push(b'\x00')
        output.push(b'\x80')
        output.push(b'\x3f')
      }
    } else {
      for _ in 0..<length {
        output.push(b'\x00')
      }
    }
  }
  if extra {
    for _ in 0..<4 {
      output.push(b'\x00')
    }
  }
  Bytes::from_array(output)
}

///|
fn fp8_artifact_digest(
  file : BytesView,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  try! @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(file)),
  )
}

///|
fn fp8_file_limits() -> @numeric_weight_file.NumericWeightFileLimits {
  try! @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=1024L * 1024L,
    max_header_bytes=256 * 1024,
    transfer_chunk_bytes=37,
    max_tensor_count=128,
  )
}

///|
fn inspect_fixture(
  root : @approved_fs.ApprovedRoot,
  name : String,
  file : BytesView,
) -> @numeric_weight_file.NumericWeightFileInspection raise @llama_fp8_weights.LlamaFp8WeightInspectionError {
  @llama_fp8_weights.inspect_full_context_file(
    root,
    fp8_file_metadata(),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
    try! @approved_fs.ApprovedRelativeLocator::new(name),
    expected_artifact_digest=fp8_artifact_digest(file),
    limits=fp8_file_limits(),
    alignment_bytes=16L,
    max_arena_bytes=1024L * 1024L,
  )
}

///|
fn inspect_paged_fixture(
  root : @approved_fs.ApprovedRoot,
  name : String,
  file : BytesView,
  batch : @llama.LlamaPagedBatchEnvelope,
) -> @numeric_weight_file.NumericWeightFileInspection raise @llama_fp8_weights.LlamaFp8WeightInspectionError {
  @llama_fp8_weights.inspect_paged_file(
    root,
    fp8_file_metadata(),
    @llama.LlamaFp8W8A8Policy::untied_v1(),
    try! @approved_fs.ApprovedRelativeLocator::new(name),
    expected_artifact_digest=fp8_artifact_digest(file),
    limits=fp8_file_limits(),
    alignment_bytes=16L,
    max_arena_bytes=1024L * 1024L,
    batch~,
  )
}

///|
async test "dense Llama FP8 inspection authenticates exact schema and digests" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-llama-fp8-exact")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let plan = fp8_file_plan()
    let file = fp8_numeric_file(plan)
    @fs.write_file("\{directory}/weights.bin", file, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let inspection = inspect_fixture(root, "weights.bin", file)
    assert_eq(inspection.identity(), plan.identity())
    assert_eq(
      inspection.numeric_schema_digest(),
      plan.numeric_binding().digest(),
    )
    assert_eq(
      inspection.layout().numeric_schema_digest(),
      Some(plan.numeric_binding().digest()),
    )
    assert_eq(
      inspection.layout().length(),
      plan.numeric_binding().tensors().length(),
    )
    assert_eq(inspection.artifact_digest(), fp8_artifact_digest(file))
    root.close()
  }
}

///|
async test "dense Llama FP8 inspection rejects aliases missing extra and overlap" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-llama-fp8-hostile")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let plan = fp8_file_plan()
    let cases : Array[
      (String, Bytes, @llama_fp8_weights.LlamaFp8WeightInspectionError),
    ] = [
      (
        "name-alias.bin",
        fp8_numeric_file(plan, alias_name_ordinal=2),
        InvalidNumericWeightFile(MissingTensor(ordinal=2)),
      ),
      (
        "dtype-alias.bin",
        fp8_numeric_file(plan, fp8_alias_ordinal=2),
        InvalidNumericWeightFile(InvalidTensorDType(ordinal=2)),
      ),
      (
        "missing.bin",
        fp8_numeric_file(plan, omit_ordinal=11),
        InvalidNumericWeightFile(TensorCountMismatch(expected=36, actual=35)),
      ),
      (
        "extra.bin",
        fp8_numeric_file(plan, extra=true),
        InvalidNumericWeightFile(TensorCountMismatch(expected=36, actual=37)),
      ),
      (
        "overlap.bin",
        fp8_numeric_file(plan, overlap_ordinal=2),
        InvalidNumericWeightFile(PayloadNotContiguous(ordinal=2)),
      ),
      (
        "layer-shape.bin",
        fp8_numeric_file(plan, wrong_shape_ordinal=11),
        InvalidNumericWeightFile(InvalidTensorShape(ordinal=11, axis=0)),
      ),
      (
        "schema.bin",
        fp8_numeric_file(plan, wrong_schema=true),
        InvalidNumericWeightFile(NumericSchemaMismatch),
      ),
    ]
    for item in cases {
      let (name, file, _) = item
      @fs.write_file("\{directory}/\{name}", file, create_mode=CreateNew)
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    for item in cases {
      let (name, file, expected) = item
      try inspect_fixture(root, name, file) catch {
        error => assert_eq(error, expected)
      } noraise {
        _ => fail("hostile dense-Llama FP8 manifest was admitted")
      }
    }
    root.close()
  }
}

///|
async test "dense Llama FP8 inspection rejects a foreign artifact digest" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-llama-fp8-digest")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let file = fp8_numeric_file(fp8_file_plan())
    @fs.write_file("\{directory}/weights.bin", file, create_mode=CreateNew)
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    try
      @llama_fp8_weights.inspect_full_context_file(
        root,
        fp8_file_metadata(),
        @llama.LlamaFp8W8A8Policy::untied_v1(),
        try! @approved_fs.ApprovedRelativeLocator::new("weights.bin"),
        expected_artifact_digest=fp8_artifact_digest(b"foreign"),
        limits=fp8_file_limits(),
        alignment_bytes=16L,
        max_arena_bytes=1024L * 1024L,
      )
    catch {
      InvalidNumericWeightFile(ArtifactDigestMismatch) => ()
      _ => fail("unexpected foreign-artifact failure")
    } noraise {
      _ => fail("foreign dense-Llama FP8 artifact digest was admitted")
    }
    root.close()
  }
}

///|
async test "paged FP8 inspection admits its exact plan and rejects identity replay" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-llama-fp8-paged")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let metadata = fp8_file_metadata()
    let policy = @llama.LlamaFp8W8A8Policy::untied_v1()
    let batch_one = @llama.LlamaPagedBatchEnvelope::single_row()
    let batch_two = try! @llama.LlamaPagedBatchEnvelope::new(2)
    let full_plan = try! @llama.build_finite_e4m3_fp8_w8a8_v1(metadata, policy)
    let paged_one = try! @llama.build_paged_finite_e4m3_fp8_w8a8_v1(
      metadata,
      policy,
      batch=batch_one,
    )
    let paged_two = try! @llama.build_paged_finite_e4m3_fp8_w8a8_v1(
      metadata,
      policy,
      batch=batch_two,
    )
    assert_true(full_plan.identity() != paged_one.identity())
    assert_true(paged_one.identity() != paged_two.identity())
    let full_file = fp8_numeric_file(full_plan)
    let paged_one_file = fp8_numeric_file(paged_one)
    let paged_two_file = fp8_numeric_file(paged_two)
    let files = [
      ("full.bin", full_file),
      ("paged-one.bin", paged_one_file),
      ("paged-two.bin", paged_two_file),
    ]
    for item in files {
      let (name, file) = item
      @fs.write_file("\{directory}/\{name}", file, create_mode=CreateNew)
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let admitted = inspect_paged_fixture(
      root, "paged-one.bin", paged_one_file, batch_one,
    )
    assert_eq(admitted.identity(), paged_one.identity())
    let replay_cases = [
      ("full.bin", full_file, batch_one),
      ("paged-one.bin", paged_one_file, batch_two),
      ("paged-two.bin", paged_two_file, batch_one),
    ]
    for item in replay_cases {
      let (name, file, batch) = item
      try inspect_paged_fixture(root, name, file, batch) catch {
        InvalidNumericWeightFile(ModelPlanMismatch) => ()
        _ => fail("unexpected paged identity replay failure")
      } noraise {
        _ => fail("foreign FP8 plan identity was replayed")
      }
    }
    try inspect_fixture(root, "paged-one.bin", paged_one_file) catch {
      InvalidNumericWeightFile(ModelPlanMismatch) => ()
      _ => fail("unexpected paged-to-full replay failure")
    } noraise {
      _ => fail("paged FP8 identity was replayed as full-context")
    }
    root.close()
  }
}

///|
async test "valid facade inputs preserve plan overflow as a typed rejection" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-llama-fp8-overflow")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let root = @approved_fs.ApprovedRoot::open_absolute(@fs.realpath(directory))
    let enormous = try! @spec.LlamaModelMetadata::from_verified_content(
      @spec.LlamaModelSpec::new(
        vocabulary_size=16,
        hidden_size=2000000000,
        intermediate_size=2000000000,
        layer_count=1,
        attention_head_count=1,
        kv_head_count=1,
        context_length=2000000000,
        rope_theta=10000.0,
        weight_dtype=BFloat16,
        kv_dtype=BFloat16,
      ),
      @spec.ContentDigest::from_sha256(
        "23456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef01",
      ),
    )
    try
      @llama_fp8_weights.inspect_paged_file(
        root,
        enormous,
        @llama.LlamaFp8W8A8Policy::untied_v1(),
        try! @approved_fs.ApprovedRelativeLocator::new("absent.bin"),
        expected_artifact_digest=fp8_artifact_digest(b"absent"),
        limits=fp8_file_limits(),
        alignment_bytes=16L,
        max_arena_bytes=1024L * 1024L,
        batch=try! @llama.LlamaPagedBatchEnvelope::new(2000000000),
      )
    catch {
      InvalidModelPlan(InvalidWorkspaceBound) => ()
      InvalidModelPlan(InvalidShape(DimensionOverflow)) => ()
      InvalidNumericWeightFile(FileUnavailable) =>
        fail("hostile dimensions reached file IO instead of plan rejection")
      _ => fail("unexpected typed plan-overflow rejection")
    } noraise {
      _ => fail("valid hostile dimensions aborted or reached file IO")
    }
    root.close()
  }
}
