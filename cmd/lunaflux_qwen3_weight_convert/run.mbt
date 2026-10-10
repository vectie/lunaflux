///|
fn join_root_locator(root : String, locator : String) -> String {
  if root.has_suffix("/") {
    root + locator
  } else {
    root + "/" + locator
  }
}

///|
fn load_qwen3_metadata(
  root : @approved_fs.ApprovedRoot,
  arguments : Qwen3WeightConvertArguments,
) -> @qwen3_spec.Qwen3ModelMetadata raise Qwen3WeightConvertError {
  let file = root.open_file(
    @approved_fs.ApprovedRelativeLocator::new(arguments.config_locator),
  ) catch {
    _ => raise ConfigAdmission
  }
  let document = file.read_immutable_snapshot(maximum_bytes=65536L) catch {
    _ => {
      file.close() catch {
        _ => ()
      }
      raise ConfigAdmission
    }
  }
  file.close() catch {
    _ => raise Cleanup
  }
  @config_reader.parse_qwen3_bytes(
    document,
    @spec.ContentDigest::from_sha256(arguments.model_content_sha256),
    max_bytes=65536,
    max_depth=16,
  ) catch {
    _ => raise ConfigAdmission
  }
}

///|
fn source_limits() -> @qwen3_weights.Qwen3SourceFileLimits raise Qwen3WeightConvertError {
  @qwen3_weights.Qwen3SourceFileLimits::new(
    max_file_bytes=4294967296L,
    compare_chunk_bytes=1024 * 1024,
    safetensors=@safetensors.HeaderLimits::new(
      max_header_bytes=4 * 1024 * 1024,
      max_tensor_count=311,
      max_rank=2,
      max_name_length=160,
      max_payload_bytes=4294967296L,
      max_materialized_tensor_bytes=4294967296L,
    ),
  ) catch {
    _ => raise SourceAdmission
  }
}

///|
fn numeric_limits(
  payload_bytes : Int64,
  header_bytes : Int,
) -> @numeric_weight_file.NumericWeightFileLimits raise Qwen3WeightConvertError {
  @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=payload_bytes + header_bytes.to_int64() + 8L,
    max_header_bytes=header_bytes,
    transfer_chunk_bytes=1024 * 1024,
    max_tensor_count=310,
  ) catch {
    _ => raise OutputAdmission
  }
}

///|
fn convert_qwen3_weight_file(
  root : @approved_fs.ApprovedRoot,
  arguments : Qwen3WeightConvertArguments,
) -> Qwen3WeightConvertReport raise Qwen3WeightConvertError {
  let metadata = load_qwen3_metadata(root, arguments)
  let source_locator = @approved_fs.ApprovedRelativeLocator::new(
    arguments.source_locator,
  ) catch {
    _ => raise SourceAdmission
  }
  let source = @qwen3_weights.inspect_source_safetensors_file(
    root,
    metadata,
    source_locator,
    expected_artifact_sha256=arguments.source_sha256,
    limits=source_limits(),
  ) catch {
    _ => raise SourceAdmission
  }
  if source.bindings().source_tensor_count() != 311 ||
    source.bindings().redundant_lm_head_tensor() is None {
    raise SourceAdmission
  }
  let batch = @qwen3.Qwen3PagedBatchEnvelope::new(arguments.max_batch_rows) catch {
    _ => raise ConversionPlan
  }
  let plan = @qwen3.build_paged(metadata, batch~) catch {
    _ => raise ConversionPlan
  }
  let conversion = prepare_qwen3_numeric_conversion(plan, source.bindings())
  root.require_absolute_identity(arguments.model_root) catch {
    _ => raise ModelRoot
  }
  publish_qwen3_numeric_file(
    join_root_locator(arguments.model_root, arguments.source_locator),
    join_root_locator(arguments.model_root, arguments.output_locator),
    conversion,
  )
  let output_locator = @approved_fs.ApprovedRelativeLocator::new(
    arguments.output_locator,
  ) catch {
    _ => raise OutputAdmission
  }
  // Compatibility field: this is an existing source-manifest identity label,
  // not a checksum of the newly copied weight payload. Publication already
  // bounds every copied region; do not read all weights again to authenticate.
  let artifact_sha256 = source.source_manifest_sha256()
  let route_sha256 = qwen3_route_manifest_sha256(
    plan,
    source.bindings(),
    artifact_sha256,
  )
  let admitted = @qwen3_weights.inspect_paged_file(
    root,
    metadata,
    output_locator,
    expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      artifact_sha256,
    ),
    limits=numeric_limits(conversion.payload_bytes, conversion.header.length()),
    alignment_bytes=256L,
    max_arena_bytes=conversion.payload_bytes + 310L * 255L,
    expected_route_manifest_sha256=route_sha256,
    batch~,
  ) catch {
    _ => raise OutputAdmission
  }
  {
    source_artifact_sha256: source.artifact_sha256(),
    source_manifest_sha256: source.source_manifest_sha256(),
    output_artifact_sha256: admitted.artifact_digest().as_hex(),
    route_manifest_sha256: admitted.route_manifest_sha256(),
  }
}

///|
fn run_qwen3_weight_convert(
  raw_arguments : ArrayView[String],
) -> Unit raise Qwen3WeightConvertError {
  let arguments = parse_qwen3_weight_convert_arguments(raw_arguments)
  let root = @approved_fs.ApprovedRoot::open_absolute(arguments.model_root) catch {
    _ => raise ModelRoot
  }
  let outcome : Result[Qwen3WeightConvertReport, Qwen3WeightConvertError] = Ok(
    convert_qwen3_weight_file(root, arguments),
  ) catch {
    error => Err(error)
  }
  let report = match outcome {
    Err(error) => {
      root.close() catch {
        _ => ()
      }
      raise error
    }
    Ok(report) => {
      root.close() catch {
        _ => raise Cleanup
      }
      report
    }
  }
  println("schema=lunaflux-qwen3-weight-convert-cli.v1")
  println("source_artifact_sha256=" + report.source_artifact_sha256)
  println("source_manifest_sha256=" + report.source_manifest_sha256)
  println("output_artifact_sha256=" + report.output_artifact_sha256)
  println("output_artifact_identity_kind=source-manifest-label")
  println("payload_checksum_verified=0")
  println("route_manifest_sha256=" + report.route_manifest_sha256)
  println("max_batch_rows=" + arguments.max_batch_rows.to_string())
  println("tensor_count=310")
  println("redundant_lm_head_omitted=1")
  println("token_step_work=0")
}
