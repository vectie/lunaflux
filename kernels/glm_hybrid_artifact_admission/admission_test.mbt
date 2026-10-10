///|
fn artifact_test_claims(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  target : @catalog.DeviceTarget,
  qualified~ : Bool,
  duplicate_ordinal? : Bool = false,
  wrong_target? : Bool = false,
  wrong_capability? : Bool = false,
  wrong_geometry? : Bool = false,
  wrong_source? : Bool = false,
  wrong_identity? : Bool = false,
  wrong_bytes? : Bool = false,
  duplicate_entry? : Bool = false,
  wrong_entry_module? : Bool = false,
  wrong_function_symbol? : Bool = false,
  wrong_launch_operands? : Bool = false,
  wrong_attention_operands? : Bool = false,
  wrong_pooling_operands? : Bool = false,
  wrong_dsa_index_operands? : Bool = false,
  wrong_sparse_attention_operands? : Bool = false,
  wrong_dsa_projection_operands? : Bool = false,
  wrong_query_a_norm_operands? : Bool = false,
  wrong_query_b_operands? : Bool = false,
  wrong_key_value_b_operands? : Bool = false,
  wrong_kv_split_norm_operands? : Bool = false,
  wrong_rotary_operands? : Bool = false,
  wrong_dense_swiglu_operands? : Bool = false,
  wrong_head_operands? : Bool = false,
  wrong_residual_operands? : Bool = false,
) -> Array[GlmHybridArtifactClaimInput] {
  let claims = []
  let digest = try! @catalog.AotArtifactDigest::from_sha256(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  )
  let foreign = artifact_test_requirements(fill='b')
  for ordinal, requirement in requirements.requirements() {
    let claim_ordinal = if duplicate_ordinal && ordinal == 1 {
      0
    } else {
      ordinal
    }
    let claim_target = if wrong_target && ordinal == 0 {
      artifact_test_target(minor=1)
    } else {
      target
    }
    let claim_capability = if wrong_capability && ordinal == 0 {
      requirements.requirements()[1].capability()
    } else {
      requirement.capability()
    }
    let claim_operation = if wrong_geometry && ordinal == 1 {
      @glm_hybrid_kernel_requirements.TokenEmbedding
    } else {
      requirement.operation()
    }
    let source = if wrong_source { foreign } else { requirements }
    let identity_ordinal = if duplicate_entry && ordinal == 1 {
      0
    } else {
      ordinal
    }
    let family_digest = if wrong_entry_module && ordinal == 0 {
      try! @catalog.AotArtifactDigest::from_sha256(
        "0000000000000000000000000000000000000000000000000000000000000000",
      )
    } else {
      digest
    }
    let family = @catalog.AotKernelFamily::new(
      module_digest=family_digest,
      family_id=try! @catalog.AotKernelFamilyId::from_stable_int(
        identity_ordinal + 1,
      ),
    )
    let entry = @catalog.AotKernelEntryPoint::new(
      family~,
      entry_point_id=try! @catalog.AotKernelEntryPointId::from_stable_int(
        identity_ordinal + 1,
      ),
    )
    let bytes = if wrong_bytes && ordinal == 0 { b"abd" } else { b"abc" }
    let (exact_function_symbol, launch, exact_operands) = artifact_test_abi(
      requirement.operation(),
      ordinal,
    )
    let function_symbol = if wrong_function_symbol && ordinal == 0 {
      "invalid-symbol"
    } else {
      exact_function_symbol
    }
    let operands = if wrong_launch_operands && ordinal == 0 {
      [
        artifact_test_operand(StepCountsInput, 20L, 16L),
        artifact_test_operand(TokenIdsInput, 8L, 4L),
        artifact_test_operand(ModelWeights, 1024L, 16L),
        artifact_test_operand(HiddenOutput, 32L, 16L),
      ]
    } else if wrong_attention_operands &&
      requirement.operation() is AttentionOutputProjection(_, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(HiddenInput, 32L, 16L),
        artifact_test_operand(ModelWeights, 126L, 16L),
        artifact_test_operand(HiddenOutput, 32L, 16L),
      ]
    } else if wrong_pooling_operands &&
      requirement.operation() is DsaIndexPooling(_, _, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(DsaIndexKeysInput, 64L, 16L),
        artifact_test_operand(DsaIndexGateScoresInput, 64L, 16L),
        artifact_test_operand(DsaIndexValidityInput, 32L, 4L),
        artifact_test_operand(ModelWeights, 16L, 16L),
        artifact_test_operand(DsaPooledKeysOutput, 32L, 16L),
        artifact_test_operand(DsaPooledRawIndicesOutput, 28L, 4L),
        artifact_test_operand(DsaPooledValidityOutput, 16L, 4L),
        artifact_test_operand(DsaPooledCountOutput, 4L, 4L),
      ]
    } else if wrong_dsa_index_operands &&
      requirement.operation() is DsaIndexCompute(_, _, _, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(QueryPositionsInput, 32L, 4L),
        artifact_test_operand(DsaIndexQueriesInput, 128L, 16L),
        artifact_test_operand(DsaIndexHeadWeightsInput, 32L, 16L),
        artifact_test_operand(DsaCandidateKeysInput, 32L, 16L),
        artifact_test_operand(DsaCandidateRawIndicesInput, 32L, 4L),
        artifact_test_operand(DsaCandidateValidityInput, 16L, 4L),
        artifact_test_operand(DsaCandidateCountInput, 4L, 4L),
        artifact_test_operand(DsaKeyValidityInput, 32L, 4L),
        artifact_test_operand(DsaSelectedIndicesOutput, 284L, 4L),
      ]
    } else if wrong_sparse_attention_operands &&
      requirement.operation() is DsaSparseAttention(_, _, _, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(QueryPositionsInput, 32L, 4L),
        artifact_test_operand(DsaKeyCountInput, 4L, 4L),
        artifact_test_operand(DsaKeyValidityInput, 32L, 4L),
        artifact_test_operand(DsaAttentionQueriesInput, 192L, 16L),
        artifact_test_operand(DsaAttentionKeysInput, 96L, 16L),
        artifact_test_operand(DsaAttentionValuesInput, 64L, 16L),
        artifact_test_operand(DsaSelectedIndicesInput, 288L, 4L),
        artifact_test_operand(DsaAttentionOutput, 126L, 16L),
      ]
    } else if wrong_dsa_projection_operands &&
      requirement.operation() is DsaQueryAProjection(_, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(HiddenInput, 32L, 16L),
        artifact_test_operand(ModelWeights, 64L, 16L),
        artifact_test_operand(HiddenOutput, 16L, 16L),
      ]
    } else if wrong_query_a_norm_operands &&
      requirement.operation() is DsaQueryARmsNorm(_, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(DsaQueryCompressedInput, 16L, 16L),
        artifact_test_operand(ModelWeights, 8L, 16L),
        artifact_test_operand(HiddenOutput, 16L, 16L),
      ]
    } else if wrong_query_b_operands &&
      requirement.operation() is DsaQueryBProjection(_, _, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(DsaQueryNormalizedInput, 16L, 16L),
        artifact_test_operand(ModelWeights, 96L, 16L),
        artifact_test_operand(HiddenOutput, 48L, 16L),
      ]
    } else if wrong_key_value_b_operands &&
      requirement.operation() is KeyValueLowRankProjection(_, _, _, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(DsaKeyValueNormalizedInput, 8L, 16L),
        artifact_test_operand(ModelWeights, 32L, 16L),
        artifact_test_operand(HiddenOutput, 32L, 16L),
      ]
    } else if wrong_kv_split_norm_operands &&
      requirement.operation() is QueryKeyRmsNorm(_) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(DsaKeyValueCompressedRopeInput, 16L, 16L),
        artifact_test_operand(ModelWeights, 4L, 16L),
        artifact_test_operand(DsaKeyValueNormalizedOutput, 8L, 16L),
      ]
    } else if wrong_rotary_operands &&
      requirement.operation() is GlmRotaryPosition(_, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(QueryPositionsInput, 8L, 4L),
        artifact_test_operand(DsaQueryHeadsOutput, 48L, 16L),
        artifact_test_operand(DsaKeyRopeSuffixOutput, 8L, 16L),
        artifact_test_operand(DsaAttentionQueriesInput, 48L, 16L),
        artifact_test_operand(DsaKeyRopeSuffixOutput, 8L, 16L),
      ]
    } else if wrong_dense_swiglu_operands &&
      requirement.operation() is DenseSwiGlu(_, _) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(HiddenInput, 32L, 16L),
        artifact_test_operand(DenseUpWeights, 256L, 16L),
        artifact_test_operand(DenseGateWeights, 256L, 16L),
        artifact_test_operand(DenseDownWeights, 256L, 16L),
        artifact_test_operand(HiddenOutput, 32L, 16L),
      ]
    } else if wrong_residual_operands &&
      requirement.operation() is MoeDecoderResidualAdd(_) {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(MoeCombinedOutput, 32L, 16L),
        artifact_test_operand(DecoderResidualInput, 32L, 16L),
        artifact_test_operand(DecoderResidualOutput, 32L, 16L),
      ]
    } else if wrong_head_operands &&
      ordinal == requirements.requirements().length() - 1 {
      [
        artifact_test_operand(StepCountsInput, 20L, 4L),
        artifact_test_operand(HiddenInput, 32L, 16L),
        artifact_test_operand(ModelWeights, 1024L, 16L),
        artifact_test_operand(HiddenOutput, 256L, 16L),
      ]
    } else {
      exact_operands
    }
    let claim = if qualified {
      GlmHybridArtifactClaimInput::new(
        ordinal=claim_ordinal,
        source_digest=source.digest(),
        source_identity=if wrong_identity && ordinal == 0 {
          foreign.source_identity()
        } else {
          source.source_identity()
        },
        target=claim_target,
        operation=claim_operation,
        capability=claim_capability,
        module_digest=digest,
        module_bytes=bytes,
        entry_point=entry,
        function_symbol~,
        launch~,
        operands~,
        declared_qualification=artifact_test_qualification(),
      )
    } else {
      GlmHybridArtifactClaimInput::new(
        ordinal=claim_ordinal,
        source_digest=source.digest(),
        source_identity=if wrong_identity && ordinal == 0 {
          foreign.source_identity()
        } else {
          source.source_identity()
        },
        target=claim_target,
        operation=claim_operation,
        capability=claim_capability,
        module_digest=digest,
        module_bytes=bytes,
        entry_point=entry,
        function_symbol~,
        launch~,
        operands~,
      )
    }
    claims.push(claim)
  }
  claims
}

///|
test "outer decoder residual operand ordering fails closed" {
  let requirements = artifact_test_requirements()
  let claims = artifact_test_claims(
    requirements,
    artifact_test_target(),
    qualified=true,
    wrong_residual_operands=true,
  )
  try
    admit(requirements, artifact_test_target(), claims, artifact_test_limits())
  catch {
    InvalidLaunchOperands(ordinal~) =>
      assert_true(
        requirements.requirements()[ordinal].operation()
        is MoeDecoderResidualAdd(_),
      )
    _ => fail("unexpected residual ABI error")
  } noraise {
    _ => fail("swapped residual and branch roles admitted")
  }
}

///|
test "all caller-declared qualification digests remain non-runnable" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let claims = artifact_test_claims(requirements, target, qualified=true)
  let first = admit(requirements, target, claims, artifact_test_limits())
  let second = admit(requirements, target, claims, artifact_test_limits())
  assert_eq(first, second)
  assert_eq(first.status(), NonRunnableWithDeclaredQualificationDigests)
  assert_eq(first.artifacts().length(), requirements.requirements().length())
  assert_eq(first.digest().as_hex().length(), 64)
  try first.require_execution_authority() catch {
    NoExecutionAuthority => ()
    _ => fail("unexpected execution-authority error")
  } noraise {
    _ => fail("artifact admission granted execution authority")
  }
}

///|
test "unqualified complete evidence remains explicitly non-runnable" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let admitted = admit(
    requirements,
    target,
    artifact_test_claims(requirements, target, qualified=false),
    artifact_test_limits(),
  )
  assert_eq(admitted.status(), NonRunnableUnqualified)
}

///|
test "missing duplicate target capability and geometry mutations fail closed" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let exact = artifact_test_claims(requirements, target, qualified=true)
  try
    admit(requirements, target, exact[1:].to_owned(), artifact_test_limits())
  catch {
    ClaimCountMismatch(expected=29, actual=28) => ()
    _ => fail("unexpected missing-claim error")
  } noraise {
    _ => fail("missing claim admitted")
  }
  let duplicate = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    duplicate_ordinal=true,
  )
  try admit(requirements, target, duplicate, artifact_test_limits()) catch {
    DuplicateClaim(ordinal=0) => ()
    _ => fail("unexpected duplicate error")
  } noraise {
    _ => fail("duplicate claim admitted")
  }
  let wrong_target = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_target=true,
  )
  try admit(requirements, target, wrong_target, artifact_test_limits()) catch {
    TargetMismatch(ordinal=0) => ()
    _ => fail("unexpected target error")
  } noraise {
    _ => fail("wrong target admitted")
  }
  let wrong_capability = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_capability=true,
  )
  try
    admit(requirements, target, wrong_capability, artifact_test_limits())
  catch {
    CapabilityMismatch(ordinal=0) => ()
    _ => fail("unexpected capability error")
  } noraise {
    _ => fail("wrong capability admitted")
  }
  let wrong_geometry = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_geometry=true,
  )
  try
    admit(requirements, target, wrong_geometry, artifact_test_limits())
  catch {
    GeometryMismatch(ordinal=1) => ()
    _ => fail("unexpected geometry error")
  } noraise {
    _ => fail("wrong geometry admitted")
  }
}

///|
test "module bytes are not checksummed but source and entrypoint contracts remain checked" {
  let requirements = artifact_test_requirements()
  let target = artifact_test_target()
  let wrong_source = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_source=true,
  )
  try admit(requirements, target, wrong_source, artifact_test_limits()) catch {
    SourceDigestMismatch(ordinal=0) => ()
    _ => fail("unexpected source error")
  } noraise {
    _ => fail("wrong source admitted")
  }
  let wrong_bytes = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_bytes=true,
  )
  let labelled = admit(
    requirements,
    target,
    wrong_bytes,
    artifact_test_limits(),
  )
  assert_eq(labelled.artifacts().length(), requirements.requirements().length())
  let wrong_identity = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_identity=true,
  )
  try
    admit(requirements, target, wrong_identity, artifact_test_limits())
  catch {
    SourceIdentityMismatch(ordinal=0) => ()
    _ => fail("unexpected source-identity error")
  } noraise {
    _ => fail("wrong source identity admitted")
  }
  let wrong_entry_module = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    wrong_entry_module=true,
  )
  try
    admit(requirements, target, wrong_entry_module, artifact_test_limits())
  catch {
    EntryPointModuleMismatch(ordinal=0) => ()
    _ => fail("unexpected entrypoint-module error")
  } noraise {
    _ => fail("wrong entrypoint module admitted")
  }
  let duplicate_entry = artifact_test_claims(
    requirements,
    target,
    qualified=true,
    duplicate_entry=true,
  )
  try
    admit(requirements, target, duplicate_entry, artifact_test_limits())
  catch {
    DuplicateEntryPoint(ordinal=1) => ()
    _ => fail("unexpected entrypoint error")
  } noraise {
    _ => fail("duplicate entrypoint admitted")
  }
  try GlmHybridDeclaredQualificationDigest::from_sha256("ABC") catch {
    InvalidDeclaredQualificationDigest => ()
    _ => fail("unexpected qualification error")
  } noraise {
    _ => fail("invalid qualification admitted")
  }
}
