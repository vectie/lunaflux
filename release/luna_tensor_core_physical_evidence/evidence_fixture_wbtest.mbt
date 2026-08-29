///|
priv struct TensorCoreEvidenceBundle {
  result : Bytes
  manifest : Bytes
  outer : Bytes
  outer_digest : LunaTensorCorePhysicalEvidenceDigest
}

///|
fn tensor_core_test_digest(
  label : String,
) -> LunaTensorCorePhysicalEvidenceDigest {
  tensor_core_evidence_sha256(@utf8.encode(label))
}

///|
fn tensor_core_test_line(
  output : StringBuilder,
  key : String,
  value : String,
) -> Unit {
  output.write_string(key)
  output.write_string("=")
  output.write_string(value)
  output.write_string("\n")
}

///|
fn tensor_core_test_manifest(
  fixture : TensorCoreEvidenceCandidateFixture,
  serial_cubin : LunaTensorCorePhysicalEvidenceDigest,
  tensor_cubin : LunaTensorCorePhysicalEvidenceDigest,
  driver : LunaTensorCorePhysicalEvidenceDigest,
  nvcc_version : LunaTensorCorePhysicalEvidenceDigest,
  ptxas_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_version : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_version : LunaTensorCorePhysicalEvidenceDigest,
  sanitizer_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_sass : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_sass : LunaTensorCorePhysicalEvidenceDigest,
  memcheck : LunaTensorCorePhysicalEvidenceDigest,
  racecheck : LunaTensorCorePhysicalEvidenceDigest,
  initcheck : LunaTensorCorePhysicalEvidenceDigest,
) -> Bytes raise {
  let empty = LunaTensorCorePhysicalEvidenceDigest::from_sha256(
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  )
  let arbitrary = tensor_core_test_digest("present evidence file")
  let entries : Array[(String, LunaTensorCorePhysicalEvidenceDigest)] = [
    ("fixture/fixture.v1", arbitrary),
    (
      "fixture/serial.canonical",
      tensor_core_evidence_read_digest(fixture.serial.digest().as_hex()),
    ),
    (
      "fixture/serial.cu",
      tensor_core_evidence_read_digest(fixture.serial.source_digest().as_hex()),
    ),
    (
      "fixture/tensor_core.canonical",
      tensor_core_evidence_read_digest(fixture.candidate.digest().as_hex()),
    ),
    (
      "fixture/tensor_core.cu",
      tensor_core_evidence_read_digest(
        fixture.candidate.source_digest().as_hex(),
      ),
    ),
    ("builds/serial/first/kernel.cubin", serial_cubin),
    ("builds/serial/second/kernel.cubin", serial_cubin),
    (
      "builds/serial/first/kernel.cu",
      tensor_core_evidence_read_digest(fixture.serial.source_digest().as_hex()),
    ),
    (
      "builds/serial/second/kernel.cu",
      tensor_core_evidence_read_digest(fixture.serial.source_digest().as_hex()),
    ),
    ("builds/tensor_core/first/kernel.cubin", tensor_cubin),
    ("builds/tensor_core/second/kernel.cubin", tensor_cubin),
    (
      "builds/tensor_core/first/kernel.cu",
      tensor_core_evidence_read_digest(
        fixture.candidate.source_digest().as_hex(),
      ),
    ),
    (
      "builds/tensor_core/second/kernel.cu",
      tensor_core_evidence_read_digest(
        fixture.candidate.source_digest().as_hex(),
      ),
    ),
    ("builds/tensor_core/resource/kernel.cubin", tensor_cubin),
    (
      "builds/tensor_core/resource/kernel.cu",
      tensor_core_evidence_read_digest(
        fixture.candidate.source_digest().as_hex(),
      ),
    ),
    ("measurements/device-identity.v1", arbitrary),
    ("measurements/driver-before.v1", driver),
    ("measurements/driver-after.v1", driver),
    ("measurements/nvcc-version-before.stdout", nvcc_version),
    ("measurements/nvcc-version-after.stdout", nvcc_version),
    ("measurements/ptxas-version-before.stdout", ptxas_version),
    ("measurements/ptxas-version-after.stdout", ptxas_version),
    ("measurements/cuobjdump-version-before.stdout", cuobjdump_version),
    ("measurements/cuobjdump-version-after.stdout", cuobjdump_version),
    ("measurements/nvdisasm-version-before.stdout", nvdisasm_version),
    ("measurements/nvdisasm-version-after.stdout", nvdisasm_version),
    ("measurements/compute-sanitizer-version-before.stdout", sanitizer_version),
    ("measurements/compute-sanitizer-version-after.stdout", sanitizer_version),
    ("measurements/cuobjdump-sass.stdout", cuobjdump_sass),
    ("measurements/nvdisasm-sass.stdout", nvdisasm_sass),
    ("measurements/cuobjdump-resources.stdout", arbitrary),
    ("measurements/memcheck-runtime.stdout", arbitrary),
    ("measurements/memcheck.log", memcheck),
    ("measurements/racecheck-runtime.stdout", arbitrary),
    ("measurements/racecheck.log", racecheck),
    ("measurements/initcheck-runtime.stdout", arbitrary),
    ("measurements/initcheck.log", initcheck),
    ("measurements/moon-build.stdout", arbitrary),
    ("measurements/export.stdout", arbitrary),
    ("builds/serial/first/compiler.stdout", arbitrary),
    ("builds/serial/first/compiler.stderr", arbitrary),
    ("builds/serial/second/compiler.stdout", arbitrary),
    ("builds/serial/second/compiler.stderr", arbitrary),
    ("builds/tensor_core/first/compiler.stdout", arbitrary),
    ("builds/tensor_core/first/compiler.stderr", arbitrary),
    ("builds/tensor_core/second/compiler.stdout", arbitrary),
    ("builds/tensor_core/second/compiler.stderr", arbitrary),
    ("builds/tensor_core/resource/compiler.stdout", arbitrary),
    ("builds/tensor_core/resource/ptxas-resource.stderr", arbitrary),
  ]
  for
    path in [
      "measurements/nvcc-version-before.stderr", "measurements/nvcc-version-after.stderr",
      "measurements/ptxas-version-before.stderr", "measurements/ptxas-version-after.stderr",
      "measurements/driver-before.stderr", "measurements/driver-after.stderr", "measurements/cuobjdump-version-before.stderr",
      "measurements/cuobjdump-version-after.stderr", "measurements/nvdisasm-version-before.stderr",
      "measurements/nvdisasm-version-after.stderr", "measurements/compute-sanitizer-version-before.stderr",
      "measurements/compute-sanitizer-version-after.stderr", "measurements/cuobjdump-sass.stderr",
      "measurements/nvdisasm-sass.stderr", "measurements/cuobjdump-resources.stderr",
      "measurements/memcheck-runtime.stderr", "measurements/racecheck-runtime.stderr",
      "measurements/initcheck-runtime.stderr", "measurements/export.stderr",
    ] {
    entries.push((path, empty))
  }
  entries.sort_by((left, right) => String::compare(left.0, right.0))
  let output = StringBuilder()
  for entry in entries {
    output.write_string(entry.1.as_hex())
    output.write_string("  ")
    output.write_string(entry.0)
    output.write_string("\n")
  }
  @utf8.encode(output.to_string())
}

///|
fn tensor_core_test_result(
  fixture : TensorCoreEvidenceCandidateFixture,
  manifest_digest : LunaTensorCorePhysicalEvidenceDigest,
  serial_cubin : LunaTensorCorePhysicalEvidenceDigest,
  tensor_cubin : LunaTensorCorePhysicalEvidenceDigest,
  driver : LunaTensorCorePhysicalEvidenceDigest,
  nvcc_version : LunaTensorCorePhysicalEvidenceDigest,
  ptxas_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_version : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_version : LunaTensorCorePhysicalEvidenceDigest,
  sanitizer_version : LunaTensorCorePhysicalEvidenceDigest,
  cuobjdump_sass : LunaTensorCorePhysicalEvidenceDigest,
  nvdisasm_sass : LunaTensorCorePhysicalEvidenceDigest,
  memcheck : LunaTensorCorePhysicalEvidenceDigest,
  racecheck : LunaTensorCorePhysicalEvidenceDigest,
  initcheck : LunaTensorCorePhysicalEvidenceDigest,
) -> Bytes {
  let output = StringBuilder()
  let fields = [
    ("schema", "lunaflux-lunatile-tensor-core-physical-campaign.v1"),
    ("outcome", "lunatile-tensor-core-physical-campaign-pass"),
    ("program_sha256", fixture.program.digest().as_hex()),
    ("serial_translation_sha256", fixture.serial.digest().as_hex()),
    ("parallel_plan_sha256", fixture.parallel.digest().as_hex()),
    ("tensor_core_candidate_sha256", fixture.candidate.digest().as_hex()),
    ("serial_source_sha256", fixture.serial.source_digest().as_hex()),
    ("tensor_core_source_sha256", fixture.candidate.source_digest().as_hex()),
    ("serial_cubin_sha256", serial_cubin.as_hex()),
    ("tensor_core_cubin_sha256", tensor_cubin.as_hex()),
    ("target", "sm_120"),
    ("device_ordinal", "0"),
    ("device_name_sha256", tensor_core_test_digest("device name").as_hex()),
    ("device_total_memory_bytes", "17179869184"),
    ("device_compute", "12.0"),
    ("device_bf16", "true"),
    ("driver_identity_sha256", driver.as_hex()),
    ("driver_record_sha256", driver.as_hex()),
    ("compiler_version", "13.1.115"),
    ("nvcc_sha256", tensor_core_test_digest("nvcc binary").as_hex()),
    ("ptxas_sha256", tensor_core_test_digest("ptxas binary").as_hex()),
    ("cuobjdump_sha256", tensor_core_test_digest("cuobjdump binary").as_hex()),
    ("nvdisasm_sha256", tensor_core_test_digest("nvdisasm binary").as_hex()),
    (
      "compute_sanitizer_sha256",
      tensor_core_test_digest("sanitizer binary").as_hex(),
    ),
    ("nvcc_version_sha256", nvcc_version.as_hex()),
    ("ptxas_version_sha256", ptxas_version.as_hex()),
    ("cuobjdump_version_sha256", cuobjdump_version.as_hex()),
    ("nvdisasm_version_sha256", nvdisasm_version.as_hex()),
    ("compute_sanitizer_version_sha256", sanitizer_version.as_hex()),
    ("sass_instruction_family", "MMA"),
    ("cuobjdump_sass_sha256", cuobjdump_sass.as_hex()),
    ("cuobjdump_sass_instruction_count", "1"),
    ("nvdisasm_sass_sha256", nvdisasm_sass.as_hex()),
    ("nvdisasm_sass_instruction_count", "1"),
    ("sass_instruction_observed", "true"),
    ("registers_per_thread", "64"),
    ("register_bound", "128"),
    ("static_shared_bytes", "0"),
    ("static_shared_bound", "0"),
    ("stack_bytes", "0"),
    ("local_bytes", "0"),
    ("spill_store_bytes", "0"),
    ("spill_load_bytes", "0"),
    ("resource_bounds_passed", "true"),
    ("numeric_case_count", "131072"),
    ("absolute_tolerance", "0.001"),
    ("relative_tolerance", "0.0001"),
    ("cpu_vs_serial_max_abs_error", "0"),
    ("cpu_vs_tensor_core_max_abs_error", "0.0009765625"),
    ("serial_vs_tensor_core_max_abs_error", "0.0009765625"),
    ("cpu_oracle", "independent-ordered-f32-v1"),
    ("serial_cuda_oracle", "passed"),
    ("memcheck_log_sha256", memcheck.as_hex()),
    ("memcheck_error_summary_count", "1"),
    ("memcheck_errors", "0"),
    ("racecheck_log_sha256", racecheck.as_hex()),
    ("racecheck_error_summary_count", "1"),
    ("racecheck_errors", "0"),
    ("initcheck_log_sha256", initcheck.as_hex()),
    ("initcheck_error_summary_count", "1"),
    ("initcheck_errors", "0"),
    (
      "cleanup_balance", "context0,stream0,allocation0,module0,function0,device_bytes0,pending0,cleanup0",
    ),
    ("physical_cuda_observed", "true"),
    ("qualification_only", "true"),
    ("manifest_bindable", "false"),
    ("promotion_authority", "absent"),
    ("evidence_files_manifest_sha256", manifest_digest.as_hex()),
  ]
  for field in fields {
    tensor_core_test_line(output, field.0, field.1)
  }
  @utf8.encode(output.to_string())
}

///|
fn tensor_core_evidence_bundle(
  fixture : TensorCoreEvidenceCandidateFixture,
) -> TensorCoreEvidenceBundle raise {
  let serial_cubin = tensor_core_test_digest("serial cubin")
  let tensor_cubin = tensor_core_test_digest("tensor core cubin")
  let driver = tensor_core_test_digest("driver report")
  let nvcc_version = tensor_core_test_digest("nvcc version")
  let ptxas_version = tensor_core_test_digest("ptxas version")
  let cuobjdump_version = tensor_core_test_digest("cuobjdump version")
  let nvdisasm_version = tensor_core_test_digest("nvdisasm version")
  let sanitizer_version = tensor_core_test_digest("sanitizer version")
  let cuobjdump_sass = tensor_core_test_digest("cuobjdump sass")
  let nvdisasm_sass = tensor_core_test_digest("nvdisasm sass")
  let memcheck = tensor_core_test_digest("memcheck log")
  let racecheck = tensor_core_test_digest("racecheck log")
  let initcheck = tensor_core_test_digest("initcheck log")
  let manifest = tensor_core_test_manifest(
    fixture, serial_cubin, tensor_cubin, driver, nvcc_version, ptxas_version, cuobjdump_version,
    nvdisasm_version, sanitizer_version, cuobjdump_sass, nvdisasm_sass, memcheck,
    racecheck, initcheck,
  )
  let result = tensor_core_test_result(
    fixture,
    tensor_core_evidence_sha256(manifest),
    serial_cubin,
    tensor_cubin,
    driver,
    nvcc_version,
    ptxas_version,
    cuobjdump_version,
    nvdisasm_version,
    sanitizer_version,
    cuobjdump_sass,
    nvdisasm_sass,
    memcheck,
    racecheck,
    initcheck,
  )
  let outer = @utf8.encode(
    tensor_core_evidence_sha256(manifest).as_hex() +
    "  FILES.sha256\n" +
    tensor_core_evidence_sha256(result).as_hex() +
    "  RESULT.txt\n",
  )
  { result, manifest, outer, outer_digest: tensor_core_evidence_sha256(outer) }
}

///|
fn tensor_core_test_replace_result(
  result : Bytes,
  key : String,
  replacement : String,
) -> Bytes raise {
  let prefix = key + "="
  let output = StringBuilder()
  let mut replaced = false
  for line in @utf8.decode(result).split("\n") {
    if line.has_prefix(prefix) {
      output.write_string(prefix + replacement + "\n")
      replaced = true
    } else if line.length() > 0 {
      output.write_string(line.to_owned())
      output.write_string("\n")
    }
  }
  if !replaced {
    abort("test result substitution key missing")
  }
  @utf8.encode(output.to_string())
}
