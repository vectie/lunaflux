///|
test "lowering rejects neutral programs and mismatched planning input" {
  let neutral = luna_cuda_test_program(semantic=false)
  try
    luna_cuda_test_lowering_with(
      neutral,
      luna_cuda_test_planning(neutral),
      luna_cuda_test_target(),
      luna_cuda_test_compiler(),
    )
  catch {
    @luna_cuda_aot.Unsupported(Program) => ()
    _ => fail("unexpected neutral-program rejection")
  } noraise {
    _ => fail("neutral program reached CUDA source lowering")
  }
  let residual = luna_cuda_test_program(width=8L)
  let other = luna_cuda_test_program(width=16L)
  try
    luna_cuda_test_lowering_with(
      residual,
      luna_cuda_test_planning(other),
      luna_cuda_test_target(),
      luna_cuda_test_compiler(),
    )
  catch {
    @luna_cuda_aot.Mismatch(PlanningInput) => ()
    _ => fail("unexpected planning-input rejection")
  } noraise {
    _ => fail("mismatched planning input reached CUDA source lowering")
  }
}

///|
test "lowering rejects unsupported target and compiler semantics" {
  try
    luna_cuda_test_lowering_with(
      luna_cuda_test_program(),
      luna_cuda_test_planning(luna_cuda_test_program()),
      luna_cuda_test_target(supports_bf16=false),
      luna_cuda_test_compiler(),
    )
  catch {
    @luna_cuda_aot.Unsupported(DeviceTarget) => ()
    _ => fail("unexpected non-BF16 rejection")
  } noraise {
    _ => fail("non-BF16 target reached CUDA source lowering")
  }
  try
    luna_cuda_test_lowering_with(
      luna_cuda_test_program(),
      luna_cuda_test_planning(luna_cuda_test_program()),
      luna_cuda_test_target(),
      luna_cuda_test_compiler(reassociate=true),
    )
  catch {
    @luna_cuda_aot.Unsupported(CompilerPolicy) => ()
    _ => fail("unexpected compiler-policy rejection")
  } noraise {
    _ => fail("reassociation reached exact residual-add lowering")
  }
}

///|
test "artifact admission rejects receipt substitution and nondeterminism" {
  let lowering = luna_cuda_test_lowering()
  let first = b"ELF-CUBIN-first"
  let second = b"ELF-CUBIN-second"
  try
    @luna_cuda_aot.admit_luna_cuda_aot_output(
      lowering,
      first,
      second,
      luna_cuda_test_receipt(lowering, first, second~),
      luna_cuda_test_limits(),
    )
  catch {
    @luna_cuda_aot.Mismatch(Determinism) => ()
    _ => fail("unexpected nondeterminism rejection")
  } noraise {
    _ => fail("different compiler outputs were admitted")
  }
  let wrong_source = @luna_cuda_aot.LunaCudaSourceDigest::from_sha256(
    "2222222222222222222222222222222222222222222222222222222222222222",
  )
  let digest = luna_cuda_test_artifact_digest(first)
  let substituted = @luna_cuda_aot.LunaCudaOfflineCompileReceipt::from_offline_builder(
    source_digest=wrong_source,
    recipe_digest=lowering.recipe_digest(),
    toolchain_digest_sha256=lowering
      .compiler_policy()
      .identity()
      .toolchain_digest_sha256(),
    driver_identity_sha256="3333333333333333333333333333333333333333333333333333333333333333",
    first_artifact_digest=digest,
    second_artifact_digest=digest,
  )
  try
    @luna_cuda_aot.admit_luna_cuda_aot_output(
      lowering,
      first,
      first,
      substituted,
      luna_cuda_test_limits(),
    )
  catch {
    @luna_cuda_aot.Mismatch(CompileReceipt) => ()
    _ => fail("unexpected substituted-receipt rejection")
  } noraise {
    _ => fail("substituted source receipt was admitted")
  }
}

///|
test "artifact admission enforces module byte limit" {
  let lowering = luna_cuda_test_lowering()
  let cubin = b"large"
  try
    @luna_cuda_aot.admit_luna_cuda_aot_output(
      lowering,
      cubin,
      cubin,
      luna_cuda_test_receipt(lowering, cubin),
      luna_cuda_test_limits(max_module_bytes=4L),
    )
  catch {
    @luna_cuda_aot.LimitExceeded(Artifact) => ()
    _ => fail("unexpected module-limit rejection")
  } noraise {
    _ => fail("oversized module was admitted")
  }
}
