///|
test "Phase5 specialization and signed capability records are byte deterministic" {
  let first = phase5_specialization()
  let second = phase5_specialization()
  assert_eq(first.digest(), second.digest())
  assert_eq(first.canonical_bytes(), second.canonical_bytes())
  let changed = phase5_specialization(optimization_level=2)
  assert_true(first.digest() != changed.digest())
  let manifest = phase5_manifest()
  let repeated = phase5_manifest()
  assert_eq(manifest.digest(), repeated.digest())
  assert_eq(manifest.signing_bytes(), repeated.signing_bytes())
}

///|
test "persisted Phase5 records require exact canonical roundtrip" {
  let offline = phase5_specialization()
  let specialization = @luna_specialization.admit_persisted_luna_specialization(
    offline.canonical_bytes(),
    offline.digest(),
    @luna_specialization.LunaSpecializationLimits::new(
      max_canonical_bytes=16384,
      max_bound_regions=4,
    ),
  )
  assert_eq(specialization, offline)
  assert_eq(
    @luna_specialization.readmit_persisted_luna_specialization(
      specialization,
      phase5_model_plan(),
      phase5_static_plan(),
      phase5_profile(),
      phase5_catalog(),
    ),
    offline,
  )
  let offline_manifest = phase5_manifest()
  let manifest = @luna_capability_manifest.admit_persisted_luna_capability_manifest(
    offline_manifest.signing_bytes(),
    offline_manifest.digest(),
    specialization,
    16384,
  )
  assert_eq(manifest, offline_manifest)
}

///|
test "persisted Phase5 records reject truncation and trailing bytes" {
  let offline = phase5_specialization()
  let canonical = offline.canonical_bytes()
  let limits = @luna_specialization.LunaSpecializationLimits::new(
    max_canonical_bytes=16384,
    max_bound_regions=4,
  )
  let truncated = Bytes::makei(canonical.length() - 1, index => canonical[index])
  try
    @luna_specialization.admit_persisted_luna_specialization(
      truncated,
      offline.digest(),
      limits,
    )
  catch {
    Invalid(CanonicalBytes) => ()
    _ => fail("unexpected truncated specialization failure")
  } noraise {
    _ => fail("truncated specialization was admitted")
  }
  let trailing = Bytes::makei(canonical.length() + 1, index => {
    if index == canonical.length() {
      b'X'
    } else {
      canonical[index]
    }
  })
  try
    @luna_specialization.admit_persisted_luna_specialization(
      trailing,
      offline.digest(),
      limits,
    )
  catch {
    Invalid(CanonicalBytes) => ()
    _ => fail("unexpected trailing specialization failure")
  } noraise {
    _ => fail("specialization with trailing bytes was admitted")
  }
  let capability = phase5_manifest()
  let capability_bytes = capability.signing_bytes()
  let truncated_capability = Bytes::makei(capability_bytes.length() - 1, index => {
    capability_bytes[index]
  })
  try
    @luna_capability_manifest.admit_persisted_luna_capability_manifest(
      truncated_capability,
      capability.digest(),
      offline,
      16384,
    )
  catch {
    Invalid(CanonicalBytes) => ()
    _ => fail("unexpected truncated capability failure")
  } noraise {
    _ => fail("truncated capability was admitted")
  }
}

///|
test "persisted Phase5 records reject substituted content identities" {
  let offline = phase5_specialization()
  let wrong_specialization = @luna_specialization.LunaSpecializationDigest::from_sha256(
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  )
  try
    @luna_specialization.admit_persisted_luna_specialization(
      offline.canonical_bytes(),
      wrong_specialization,
      @luna_specialization.LunaSpecializationLimits::new(
        max_canonical_bytes=16384,
        max_bound_regions=4,
      ),
    )
  catch {
    Mismatch(Digest) => ()
    _ => fail("unexpected specialization digest substitution failure")
  } noraise {
    _ => fail("substituted specialization digest was admitted")
  }
  let capability = phase5_manifest()
  let wrong_capability = @luna_capability_manifest.LunaCapabilityManifestDigest::from_sha256(
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
  )
  try
    @luna_capability_manifest.admit_persisted_luna_capability_manifest(
      capability.signing_bytes(),
      wrong_capability,
      offline,
      16384,
    )
  catch {
    Mismatch(Digest) => ()
    _ => fail("unexpected capability digest substitution failure")
  } noraise {
    _ => fail("substituted capability digest was admitted")
  }
}

///|
test "offline specialization rejects a neutral graph without semantic authority" {
  try phase5_specialization(program_semantic=false) catch {
    @luna_specialization.Mismatch(OperationBinding) => ()
    _ => fail("unexpected neutral-program specialization failure")
  } noraise {
    _ => fail("neutral LunaTile graph was specialized")
  }
}

///|
test "offline specialization rejects residual semantics for another operation" {
  try phase5_specialization(operation_id=3) catch {
    @luna_specialization.Mismatch(OperationBinding) => ()
    _ => fail("unexpected cross-operation specialization failure")
  } noraise {
    _ => fail("residual semantics were substituted for rotary embedding")
  }
}

///|
test "offline specialization binds ordered ModelPlan and profile roles" {
  try phase5_specialization(swap_plan_residual_inputs=true) catch {
    @luna_specialization.Mismatch(Identity) => ()
    _ => fail("unexpected ordered profile-role failure")
  } noraise {
    _ => fail("profile roles were substituted across ModelPlan order")
  }
}

///|
test "BF16 specialization rejects a neutral non-BF16 LunaTile program" {
  try
    phase5_specialization(program_dtype=Float16, program_semantic=false)
  catch {
    @luna_specialization.Unsupported(Quantization) => ()
    _ => fail("unexpected specialization dtype failure")
  } noraise {
    _ => fail("non-BF16 LunaTile program was specialized for BF16")
  }
}
