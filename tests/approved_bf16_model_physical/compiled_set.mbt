///|
fn compiled_value(document : String, key : String) -> String? {
  let prefix = key + "="
  let mut found : String? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) {
        return None
      }
      found = Some(line[prefix.length():].to_owned())
    }
  }
  found
}

///|
fn require_compiled_value(
  document : String,
  key : String,
) -> String raise ApprovedBf16PhysicalError {
  guard compiled_value(document, key) is Some(value) else {
    raise InvalidCompiledSet("missing or duplicate field: \{key}")
  }
  value
}

///|
fn require_compiled_sha256(
  value : String,
  message : String,
) -> Unit raise ApprovedBf16PhysicalError {
  if value.length() != 64 {
    raise InvalidCompiledSet(message)
  }
  for byte in @utf8.encode(value) {
    if !((byte >= b'0' && byte <= b'9') || (byte >= b'a' && byte <= b'f')) {
      raise InvalidCompiledSet(message)
    }
  }
}

///|
fn parse_physical_compiler_component(
  value : String,
  allow_zero : Bool,
) -> UInt raise ApprovedBf16PhysicalError {
  if value.is_empty() || value.length() > 6 {
    raise InvalidInput("compiler version component is invalid")
  }
  let mut result = 0U
  for unit in value {
    guard unit is ('0'..='9') else {
      raise InvalidInput("compiler version component is invalid")
    }
    result = result * 10U + (unit.to_int() - '0'.to_int()).reinterpret_as_uint()
  }
  if result == 0U && !allow_zero {
    raise InvalidInput("compiler major version must be positive")
  }
  result
}

///|
fn candidate_operation_row(
  document : String,
  operation_id : Int,
) -> FixedArray[String] raise ApprovedBf16PhysicalError {
  let prefix = "operation=\{operation_id},"
  let mut found : FixedArray[String]? = None
  for line in document.split("\n") {
    if line.has_prefix(prefix) {
      if found is Some(_) {
        raise InvalidCompiledSet("duplicate compiled operation")
      }
      found = Some(
        FixedArray::from_array(
          line[10:].split(",").map(value => value.to_owned()).to_array(),
        ),
      )
    }
  }
  guard found is Some(fields) && fields.length() == 5 else {
    raise InvalidCompiledSet("compiled operation row is malformed")
  }
  fields
}

///|
async fn approved_offline_compilations(
  compiled_root : String,
  candidate_set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  expected_toolchain_sha256 : String,
  expected_driver_identity_sha256 : String,
) -> Array[@luna_bf16_kernel_producer.LunaBf16OfflineCompilation] {
  let canonical = @fs.realpath(compiled_root)
  if canonical != compiled_root {
    raise InvalidCompiledSet("compiled root must be canonical and absolute")
  }
  let compiled_bytes = @fs.read_file("\{compiled_root}/compiled-set.v1").binary()
  let compiled = @utf8.decode(compiled_bytes)
  if require_compiled_value(compiled, "schema") !=
    "lunaflux-bf16-compiled-kernel-set.v1" ||
    require_compiled_value(compiled, "candidate_set_sha256") !=
    candidate_set.candidate_set_sha256() ||
    require_compiled_value(compiled, "candidate_inventory_sha256") !=
    candidate_set.inventory_sha256() ||
    require_compiled_value(compiled, "toolchain_sha256") !=
    expected_toolchain_sha256 ||
    require_compiled_value(compiled, "driver_identity_sha256") !=
    expected_driver_identity_sha256 ||
    require_compiled_value(compiled, "target") != "sm_120" ||
    require_compiled_value(compiled, "operation_count") !=
    candidate_set.operation_count().to_string() {
    raise InvalidCompiledSet("compiled-set identity mismatch")
  }
  let compilations : Array[
    @luna_bf16_kernel_producer.LunaBf16OfflineCompilation,
  ] = []
  for index, candidate in candidate_set.candidates() {
    let row = candidate_operation_row(compiled, index)
    guard candidate_set.candidate_key(index) is Some(key) else {
      raise InvalidCompiledSet("candidate key is missing")
    }
    if row[0] != index.to_string() ||
      row[1] != key ||
      row[2] != candidate.family_name() {
      raise InvalidCompiledSet("compiled operation identity mismatch")
    }
    let module_sha = row[3]
    let receipt_sha = row[4]
    require_compiled_sha256(module_sha, "compiled module digest is invalid")
    require_compiled_sha256(receipt_sha, "compile receipt digest is invalid")
    let receipt_bytes = @fs.read_file(
      "\{compiled_root}/receipts/\{key}.receipt.v1",
    ).binary()
    if approved_sha256(receipt_bytes) != receipt_sha {
      raise InvalidCompiledSet("compile receipt digest mismatch")
    }
    let receipt_text = @utf8.decode(receipt_bytes)
    let first_sha = require_compiled_value(
      receipt_text, "first_artifact_sha256",
    )
    let second_sha = require_compiled_value(
      receipt_text, "second_artifact_sha256",
    )
    if require_compiled_value(receipt_text, "schema") !=
      "lunaflux-bf16-offline-compile-receipt.v1" ||
      require_compiled_value(receipt_text, "candidate_set_sha256") !=
      candidate_set.candidate_set_sha256() ||
      require_compiled_value(receipt_text, "operation_id") != index.to_string() ||
      require_compiled_value(receipt_text, "family") != candidate.family_name() ||
      require_compiled_value(receipt_text, "source_sha256") !=
      candidate.source_sha256() ||
      require_compiled_value(receipt_text, "recipe_sha256") !=
      candidate.recipe_sha256() ||
      require_compiled_value(receipt_text, "target") != "sm_120" ||
      require_compiled_value(receipt_text, "toolchain_sha256") !=
      expected_toolchain_sha256 ||
      require_compiled_value(receipt_text, "driver_identity_sha256") !=
      expected_driver_identity_sha256 ||
      require_compiled_value(receipt_text, "deterministic") != "1" ||
      first_sha != module_sha ||
      second_sha != module_sha ||
      require_compiled_value(receipt_text, "module_relative_path") !=
      "sha256/\{module_sha}.cubin" {
      raise InvalidCompiledSet("compile receipt binding mismatch")
    }
    let module_bytes = @fs.read_file(
      "\{compiled_root}/sha256/\{module_sha}.cubin",
    ).binary()
    if approved_sha256(module_bytes) != module_sha {
      raise InvalidCompiledSet("compiled CUBIN digest mismatch")
    }
    let receipt = @luna_kernel_bundle.LunaDeterministicCompileReceipt::from_offline_builder(
      source_digest_sha256=candidate.source_sha256(),
      recipe_digest_sha256=candidate.recipe_sha256(),
      toolchain_digest_sha256=expected_toolchain_sha256,
      driver_identity_sha256=expected_driver_identity_sha256,
      first_artifact_digest_sha256=first_sha,
      second_artifact_digest_sha256=second_sha,
    )
    compilations.push(
      @luna_bf16_kernel_producer.LunaBf16OfflineCompilation::new(
        candidate~,
        first_module=module_bytes,
        second_module=module_bytes,
        receipt~,
      ),
    )
  }
  compilations
}
