///|
fn phase4_correctness_name(index : Int) -> String {
  "correctness/trial-" + index.to_string() + ".correctness.v1"
}

///|
fn phase4_require_field(
  lines : ArrayView[StringView],
  index : Int,
  key : String,
) -> String raise Phase4PrefixGateError {
  if index < 0 || index >= lines.length() {
    raise Phase4InvalidCorrectness
  }
  let prefix = key + "="
  if !lines[index].has_prefix(prefix) {
    raise Phase4InvalidCorrectness
  }
  "\{lines[index][prefix.length():]}"
}

///|
fn phase4_correctness_digest(
  bytes : Bytes,
  declaration_sha256 : String,
  capture_sha256 : String,
  trial_index : Int,
) -> @evidence.BenchmarkDigest raise Phase4PrefixGateError {
  let document = @utf8.decode(bytes) catch {
    _ => raise Phase4InvalidCorrectness
  }
  let lines = document.split("\n").to_array()
  if lines.length() != 7 ||
    lines[6] != "" ||
    phase4_require_field(lines, 0, "schema") !=
    "lunaflux.external-correctness-observation.v1" ||
    phase4_require_field(lines, 1, "declaration_sha256") != declaration_sha256 ||
    phase4_require_field(lines, 2, "trial_index") != trial_index.to_string() ||
    phase4_require_field(lines, 3, "capture_sha256") != capture_sha256 ||
    phase4_require_field(lines, 5, "correctness_passed") != "1" {
    raise Phase4InvalidCorrectness
  }
  ignore(
    @evidence.BenchmarkDigest::from_sha256(
      phase4_require_field(lines, 4, "reference_sha256"),
    ) catch {
      _ => raise Phase4InvalidCorrectness
    },
  )
  @evidence.BenchmarkDigest::from_sha256(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
  ) catch {
    _ => raise Phase4InvalidCorrectness
  }
}

///|
fn phase4_inventory_entry(
  inventory_lines : ArrayView[StringView],
  relative : String,
  digest : @evidence.BenchmarkDigest,
) -> Unit raise Phase4PrefixGateError {
  let expected = digest.as_hex() + "  " + relative
  for line in inventory_lines {
    if line == expected {
      return
    }
  }
  raise Phase4InvalidCorrectness
}

///|
async fn phase4_load_correctness(
  root : String,
  declaration_sha256 : String,
  captures : ArrayView[Bytes],
  expected_inventory_sha256 : String,
) -> Array[@evidence.BenchmarkDigest] raise Phase4PrefixGateError {
  let inventory = read_regular_capture(root, "correctness.files.sha256") catch {
    _ => raise Phase4InvalidCorrectness
  }
  if inventory.length() <= 0 ||
    inventory.length() > PHASE4_MAX_CORRECTNESS_INVENTORY_BYTES {
    raise Phase4InvalidCorrectness
  }
  let inventory_digest = @crypto.bytes_to_hex_string(@crypto.sha256(inventory))
  if inventory_digest != expected_inventory_sha256 {
    raise Phase4InvalidCorrectness
  }
  let inventory_document = @utf8.decode(inventory) catch {
    _ => raise Phase4InvalidCorrectness
  }
  let inventory_lines = inventory_document.split("\n").to_array()
  if inventory_lines.length() != PHASE4_CORRECTNESS_TRIALS + 1 ||
    inventory_lines[PHASE4_CORRECTNESS_TRIALS] != "" ||
    captures.length() != PHASE4_CORRECTNESS_TRIALS {
    raise Phase4InvalidCorrectness
  }
  let correctness : Array[@evidence.BenchmarkDigest] = []
  for index in 0..<PHASE4_CORRECTNESS_TRIALS {
    let relative = phase4_correctness_name(index)
    let bytes = read_regular_capture(root, relative) catch {
      _ => raise Phase4InvalidCorrectness
    }
    if bytes.length() <= 0 || bytes.length() > PHASE4_MAX_CORRECTNESS_BYTES {
      raise Phase4InvalidCorrectness
    }
    let capture_sha256 = @crypto.bytes_to_hex_string(
      @crypto.sha256(captures[index]),
    )
    let digest = phase4_correctness_digest(
      bytes, declaration_sha256, capture_sha256, index,
    )
    phase4_inventory_entry(inventory_lines, relative, digest)
    for prior in correctness {
      if prior == digest {
        raise Phase4InvalidCorrectness
      }
    }
    correctness.push(digest)
  }
  correctness
}
