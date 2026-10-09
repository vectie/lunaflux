///|
fn compile_evidence_test_compiler(
  toolchain_fill? : Char = 'a',
) -> @luna_specialization.LunaCompilerPolicy {
  try! @luna_specialization.LunaCompilerPolicy::new(
    identity=try! @luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256=String::make(64, toolchain_fill),
      major=12,
      minor=9,
      patch=1,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
}

///|
fn compile_evidence_test_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn compile_evidence_test_limits(
  maximum_module_bytes? : Int64 = 1024L,
  maximum_receipt_bytes? : Int = 4096,
) -> CudaOfflineCompileEvidenceLimits {
  try! CudaOfflineCompileEvidenceLimits::new(
    maximum_module_bytes~,
    maximum_receipt_bytes~,
    maximum_symbol_bytes=128,
  )
}

///|
fn compile_evidence_test_receipt(
  module_bytes? : Bytes = b"deterministic-cubin",
  source? : String = String::make(64, 'b'),
  recipe? : String = String::make(64, 'c'),
  compiler? : @luna_specialization.LunaCompilerPolicy = compile_evidence_test_compiler(),
  driver? : String = String::make(64, 'd'),
  target? : @catalog.DeviceTarget = compile_evidence_test_target(),
  symbol? : String = "lunaflux_test_kernel_v1",
) -> Bytes {
  expected_compile_receipt(
    source,
    recipe,
    compiler,
    driver,
    target,
    symbol,
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)),
  )
}

///|
fn compile_evidence_test_admit(
  first? : Bytes = b"deterministic-cubin",
  second? : Bytes = b"deterministic-cubin",
  source? : String = String::make(64, 'b'),
  recipe? : String = String::make(64, 'c'),
  compiler? : @luna_specialization.LunaCompilerPolicy = compile_evidence_test_compiler(),
  driver? : String = String::make(64, 'd'),
  target? : @catalog.DeviceTarget = compile_evidence_test_target(),
  symbol? : String = "lunaflux_test_kernel_v1",
  receipt? : Bytes = compile_evidence_test_receipt(),
  expected_receipt? : String = @crypto.bytes_to_hex_string(
    @crypto.sha256(receipt),
  ),
  limits? : CudaOfflineCompileEvidenceLimits = compile_evidence_test_limits(),
) -> CudaOfflineCompileEvidence raise CudaOfflineCompileEvidenceError {
  admit_deterministic_cuda_output(
    source_digest_sha256=source,
    recipe_digest_sha256=recipe,
    compiler~,
    driver_identity_sha256=driver,
    target~,
    function_symbol=symbol,
    first_module=first,
    second_module=second,
    receipt_bytes=receipt,
    expected_receipt_digest_sha256=expected_receipt,
    limits~,
  )
}

///|
test "deterministic CUDA receipt retains exact inert self-consistency" {
  let first = compile_evidence_test_admit()
  let repeated = compile_evidence_test_admit()
  assert_eq(first.status(), InertSelfConsistent)
  assert_eq(first.source_digest_sha256(), String::make(64, 'b'))
  assert_eq(first.recipe_digest_sha256(), String::make(64, 'c'))
  assert_eq(first.compiler_policy(), compile_evidence_test_compiler())
  assert_eq(first.driver_identity_sha256(), String::make(64, 'd'))
  assert_eq(first.target(), compile_evidence_test_target())
  assert_eq(first.function_symbol(), "lunaflux_test_kernel_v1")
  assert_eq(
    first.module_digest().sha256(),
    @crypto.bytes_to_hex_string(@crypto.sha256(b"deterministic-cubin")),
  )
  assert_eq(first.module_byte_count(), b"deterministic-cubin".length())
  assert_eq(first.digest(), repeated.digest())
  assert_eq(first.digest().as_hex().length(), 64)
  try first.require_producer_authority() catch {
    ProducerAuthorityUnavailable => ()
    _ => fail("unexpected producer-authority rejection")
  } noraise {
    _ => fail("self-consistent receipt fabricated producer authority")
  }
  try first.require_compiler_execution_proof() catch {
    CompilerExecutionProofUnavailable => ()
    _ => fail("unexpected compiler-proof rejection")
  } noraise {
    _ => fail("self-consistent receipt fabricated compiler execution")
  }
  try first.require_execution_authority() catch {
    ExecutionAuthorityUnavailable => ()
    _ => fail("unexpected execution-authority rejection")
  } noraise {
    _ => fail("compile evidence gained execution authority")
  }
}

///|
test "digest and symbol identities reject malformed inputs" {
  let malformed = [
    (String::make(63, 'b'), String::make(64, 'c'), String::make(64, 'd')),
    (String::make(64, 'b'), String::make(64, 'z'), String::make(64, 'd')),
    (String::make(64, 'b'), String::make(64, 'c'), String::make(64, 'D')),
  ]
  for index, entry in malformed {
    let (source, recipe, driver) = entry
    try compile_evidence_test_admit(source~, recipe~, driver~) catch {
      InvalidDigest(Source) if index == 0 => ()
      InvalidDigest(Recipe) if index == 1 => ()
      InvalidDigest(Driver) if index == 2 => ()
      _ => fail("unexpected malformed compile identity rejection")
    } noraise {
      _ => fail("malformed compile identity was admitted")
    }
  }
  try compile_evidence_test_admit(symbol="9invalid") catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected compile symbol rejection")
  } noraise {
    _ => fail("invalid compile symbol was admitted")
  }
}

///|
test "module and receipt substitution fail closed" {
  try compile_evidence_test_admit(first=b"", second=b"") catch {
    EmptyModule => ()
    _ => fail("unexpected empty-module rejection")
  } noraise {
    _ => fail("empty deterministic modules were admitted")
  }
  try
    compile_evidence_test_admit(
      first=b"deterministic-cubin",
      second=b"substituted-cubin",
    )
  catch {
    NondeterministicModule => ()
    _ => fail("unexpected nondeterministic-module rejection")
  } noraise {
    _ => fail("different build outputs were admitted")
  }
  try
    compile_evidence_test_admit(
      first=b"too-large",
      second=b"too-large",
      receipt=compile_evidence_test_receipt(module_bytes=b"too-large"),
      limits=compile_evidence_test_limits(maximum_module_bytes=4L),
    )
  catch {
    ModuleTooLarge => ()
    _ => fail("unexpected module-limit rejection")
  } noraise {
    _ => fail("oversized modules were admitted")
  }
  try compile_evidence_test_admit(receipt=b"") catch {
    EmptyReceipt => ()
    _ => fail("unexpected empty-receipt rejection")
  } noraise {
    _ => fail("empty compile receipt was admitted")
  }
  try
    compile_evidence_test_admit(expected_receipt=String::make(64, 'e'))
  catch {
    ReceiptDigestMismatch => ()
    _ => fail("unexpected receipt-digest rejection")
  } noraise {
    _ => fail("foreign receipt digest was admitted")
  }
  let substituted = b"schema=substituted\n"
  try
    compile_evidence_test_admit(
      receipt=substituted,
      expected_receipt=@crypto.bytes_to_hex_string(@crypto.sha256(substituted)),
    )
  catch {
    ReceiptContentMismatch => ()
    _ => fail("unexpected receipt-content rejection")
  } noraise {
    _ => fail("substituted canonical receipt was admitted")
  }
}

///|
test "receipt binds source recipe compiler driver and target fields" {
  let receipt = compile_evidence_test_receipt()
  let receipt_digest = @crypto.bytes_to_hex_string(@crypto.sha256(receipt))
  for index in 0..<5 {
    try
      compile_evidence_test_admit(
        source=if index == 0 {
          String::make(64, 'e')
        } else {
          String::make(64, 'b')
        },
        recipe=if index == 1 {
          String::make(64, 'e')
        } else {
          String::make(64, 'c')
        },
        compiler=if index == 2 {
          compile_evidence_test_compiler(toolchain_fill='e')
        } else {
          compile_evidence_test_compiler()
        },
        driver=if index == 3 {
          String::make(64, 'e')
        } else {
          String::make(64, 'd')
        },
        target=if index == 4 {
          compile_evidence_test_target(minor=1)
        } else {
          compile_evidence_test_target()
        },
        receipt~,
        expected_receipt=receipt_digest,
      )
    catch {
      ReceiptContentMismatch => ()
      _ => fail("unexpected receipt-field rejection")
    } noraise {
      _ => fail("substituted compile receipt field was admitted")
    }
  }
}
