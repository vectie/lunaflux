///|
fn record_line(record : @strategy.ProjectionAutotuneRecord) -> String {
  let key = record.key()
  let shape = key.shape()
  let selected = key.strategy()
  let (input, output, intermediate) = shape.dimensions()
  let (rows, columns, reduction) = selected.tile().dimensions()
  let family = match shape.family() {
    Qkv => 1
    Dense => 2
    GatedMlp => 3
    LanguageModelHead => 4
  }
  let numeric = match key.numeric_type() {
    ProjectionBFloat16 => 1
    ProjectionFloat16 => 2
    ProjectionFloat32 => 3
    ProjectionSignedInt8 => 4
    ProjectionFloat8 => 5
  }
  let kernel = match selected.kernel_class() {
    ScalarTile => 1
    SubgroupGemv => 2
    MatrixTile => 3
  }
  let residency = if selected.input_residency() == SelectedRowTileResidency {
    2
  } else {
    1
  }
  // Phase and workload class are uniquely derived by the typed row bucket.
  "record\t\{family}\t\{numeric}\t\{input}\t\{output}\t\{intermediate}\t\{selected.stable_id()}\t\{selected.row_bucket()}\t\{kernel}\t\{rows}\t\{columns}\t\{reduction}\t\{selected.output_tiles_per_workgroup()}\t\{residency}\t\{record.median_latency_ns()}\t\{record.sample_count()}"
}

///|
pub fn ProjectionTuningSnapshot::new(
  scope : ProjectionTuningScope,
  records : ArrayView[@strategy.ProjectionAutotuneRecord],
) -> ProjectionTuningSnapshot raise ProjectionTuningError {
  if records.is_empty() || records.length() > 4096 {
    raise InvalidSnapshot
  }
  let lines : Array[String] = []
  for index, record in records {
    let key = record.key()
    if key.backend() != scope.backend || key.target() != scope.architecture {
      raise ScopeMismatch
    }
    for previous in 0..<index {
      let other = records[previous].key()
      if key.shape() == other.shape() &&
        key.numeric_type() == other.numeric_type() &&
        key.strategy().row_bucket() == other.strategy().row_bucket() &&
        key.strategy().stable_id() == other.strategy().stable_id() {
        raise InvalidSnapshot
      }
    }
    lines.push(record_line(record))
  }
  lines.sort()
  let builder = StringBuilder()
  builder <+ "luna-projection-tuning-v1\n"
  builder <+
    "scope\t\{scope.backend.stable_name()}\t\{scope.architecture.stable_name()}\t\{scope.device_identity}\t\{scope.toolchain_sha256}\n"
  for line in lines {
    builder <+ "\{line}\n"
  }
  {
    scope,
    records: ReadOnlyArray::from_array(records),
    canonical: @utf8.encode(builder.to_string()),
  }
}

///|
fn decimal(value : StringView) -> Int64 raise ProjectionTuningError {
  if value.is_empty() ||
    value.length() > 19 ||
    (value.length() > 1 && value[0] == '0') {
    raise InvalidSnapshot
  }
  let mut result = 0L
  for c in value {
    guard c is ('0'..='9') else { raise InvalidSnapshot }
    let digit = (c.to_int() - '0'.to_int()).to_int64()
    if result > (9223372036854775807L - digit) / 10L {
      raise InvalidSnapshot
    }
    result = result * 10L + digit
  }
  result
}

///|
fn integer(value : StringView) -> Int raise ProjectionTuningError {
  let value = decimal(value)
  if value > 2147483647L {
    raise InvalidSnapshot
  }
  value.to_int()
}

///|
fn parse_record(
  line : StringView,
  scope : ProjectionTuningScope,
) -> @strategy.ProjectionAutotuneRecord raise ProjectionTuningError {
  guard line.split("\t").to_array()
    is [
      "record",
      family,
      numeric,
      input,
      output,
      intermediate,
      id,
      bucket,
      kernel,
      rows,
      columns,
      reduction,
      groups,
      residency,
      latency,
      samples,
    ] else {
    raise InvalidSnapshot
  }
  let family = match family {
    "1" => @strategy.Qkv
    "2" => Dense
    "3" => GatedMlp
    "4" => LanguageModelHead
    _ => raise InvalidSnapshot
  }
  let numeric = match numeric {
    "1" => @strategy.ProjectionBFloat16
    "2" => ProjectionFloat16
    "3" => ProjectionFloat32
    "4" => ProjectionSignedInt8
    "5" => ProjectionFloat8
    _ => raise InvalidSnapshot
  }
  let kernel = match kernel {
    "1" => @strategy.ScalarTile
    "2" => SubgroupGemv
    "3" => MatrixTile
    _ => raise InvalidSnapshot
  }
  let residency = match residency {
    "1" => @strategy.ReductionTileResidency
    "2" => SelectedRowTileResidency
    _ => raise InvalidSnapshot
  }
  let bucket = integer(bucket)
  @strategy.ProjectionAutotuneRecord::new(
    key=@strategy.ProjectionAutotuneKey::new(
      backend=scope.backend,
      target=scope.architecture,
      numeric_type=numeric,
      shape=@strategy.ProjectionShape::new(
        family,
        input_width=integer(input),
        output_width=integer(output),
        intermediate_width=integer(intermediate),
      ),
      strategy=@strategy.ProjectionStrategy::new(
        stable_id=integer(id),
        phase=if bucket == 1 { Decode } else { Prefill },
        kernel_class=kernel,
        row_bucket=bucket,
        tile=@strategy.ProjectionTile::new(
          rows=integer(rows),
          columns=integer(columns),
          reduction=integer(reduction),
        ),
        output_tiles_per_workgroup=integer(groups),
        input_residency=residency,
      ),
    ),
    median_latency_ns=decimal(latency),
    sample_count=integer(samples),
  ) catch {
    _ => raise InvalidSnapshot
  }
}

///|
pub fn parse_projection_tuning_snapshot(
  bytes : Bytes,
) -> ProjectionTuningSnapshot raise ProjectionTuningError {
  if bytes.length() > 1048576 {
    raise InvalidSnapshot
  }
  let text = @utf8.decode(bytes) catch { _ => raise InvalidSnapshot }
  guard text.split("\n").to_array()
    is ["luna-projection-tuning-v1", scope_line, .. rows, ""] else {
    raise InvalidSnapshot
  }
  guard scope_line.split("\t").to_array()
    is ["scope", backend, architecture, device, toolchain] else {
    raise InvalidSnapshot
  }
  let scope = ProjectionTuningScope::new(
    backend=@strategy.ProjectionBackendId::new(backend.to_owned()),
    architecture=@strategy.ProjectionTargetId::new(architecture.to_owned()),
    device_identity=device.to_owned(),
    toolchain_sha256=toolchain.to_owned(),
  ) catch {
    _ => raise InvalidSnapshot
  }
  let records = rows.map(row => parse_record(row, scope))
  let snapshot = ProjectionTuningSnapshot::new(scope, records)
  if snapshot.canonical != bytes {
    raise InvalidSnapshot
  }
  snapshot
}
