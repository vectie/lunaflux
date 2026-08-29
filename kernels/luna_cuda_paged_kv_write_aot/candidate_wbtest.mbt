///|
test "positioned paged KV write source and recipe are deterministic and inert" {
  let fixture = paged_kv_write_test_fixture()
  let first = paged_kv_write_test_candidate(fixture)
  let second = paged_kv_write_test_candidate(fixture)
  assert_eq(first.source_digest(), second.source_digest())
  assert_eq(first.recipe_digest(), second.recipe_digest())
  assert_eq(first.candidate_digest(), second.candidate_digest())
  assert_false(first.manifest_bindable())
  assert_false(first.promotion_authority())
  assert_eq(first.kernel_abi(), PositionedQkvPagedWriteBf16V1)
  let source = @utf8.decode(first.source_bytes())
  assert_true(source.contains("const __nv_bfloat16* positioned_qkv"))
  assert_true(source.contains("position >= sequence_lengths[row]"))
  assert_true(source.contains("page_table_offsets[r + 1] > table_cells"))
  assert_true(
    source.contains("key_cache[cache_index] = positioned_qkv[qkv_base +"),
  )
  assert_true(
    source.contains("value_cache[cache_index] = positioned_qkv[qkv_base +"),
  )
  assert_true(source.contains("atomicAdd(dispatch_canary, 22321U)"))
  assert_true(
    source.contains("(long long)rows != (long long)prefill + (long long)decode"),
  )
  assert_true(
    source.contains("logical_page < 0 || logical_page >= end - begin"),
  )
  assert_false(source.contains("attention"))
  let recipe = @utf8.decode(first.recipe_bytes())
  assert_true(
    recipe.contains("fallback_family=monolithic-paged-write-and-attend"),
  )
  assert_true(recipe.contains("manifest_bindable=false"))
  let exported = export_positioned_paged_kv_write_qualification(first)
  assert_eq(exported.source_digest(), first.source_digest())
  assert_false(exported.manifest_bindable())
  assert_false(exported.physical_evidence())
  assert_false(exported.promotion_authority())
}

///|
test "positioned paged KV write rejects arbitrary activation and operand substitution" {
  let fixture = paged_kv_write_test_fixture()
  try
    paged_kv_write_test_candidate(
      fixture,
      input=PositionedQkvActivation::new(qkv=@plan.ActivationRef::new(100)),
    )
  catch {
    Mismatch(Identity) => ()
    _ => fail("unexpected positioned activation rejection")
  } noraise {
    _ => fail("arbitrary positioned activation reached candidate")
  }
  let changed = fixture.operands.copy()
  changed[7] = changed[6]
  try paged_kv_write_test_candidate(fixture, operands=changed[:]) catch {
    Mismatch(OperandOrder) => ()
    _ => fail("unexpected operand substitution rejection")
  } noraise {
    _ => fail("substituted operand reached candidate")
  }
}

///|
test "positioned paged KV write rejects cross-layout replay" {
  let fixture = paged_kv_write_test_fixture()
  let replay = @device_layout.plan(
    fixture.plan.identity(),
    fixture.plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=8,
      total_page_count=64,
      max_arena_bytes=67108864L,
    ),
  )
  try paged_kv_write_test_candidate(fixture, layout=replay) catch {
    Mismatch(Layout | Fallback | OperandSize) => ()
    _ => fail("unexpected layout replay rejection")
  } noraise {
    _ => fail("cross-layout replay reached candidate")
  }
}

///|
test "positioned paged KV write rejects target and compiler substitution" {
  let fixture = paged_kv_write_test_fixture()
  let target = @catalog.DeviceTarget::new(
    compute_major=8,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  try paged_kv_write_test_candidate(fixture, target~) catch {
    Mismatch(Fallback) => ()
    _ => fail("unexpected target substitution rejection")
  } noraise {
    _ => fail("cross-target candidate replay was admitted")
  }
  try
    paged_kv_write_test_candidate(
      fixture,
      compiler=paged_kv_write_test_compiler(reassociate=true),
    )
  catch {
    Unsupported(Compiler) => ()
    _ => fail("unexpected compiler substitution rejection")
  } noraise {
    _ => fail("unsupported compiler policy was admitted")
  }
}

///|
test "compiled positioned paged KV write binds deterministic receipt only" {
  let candidate = paged_kv_write_test_candidate(paged_kv_write_test_fixture())
  let module_ = b"fixture-only-positioned-kv-write-cubin"
  let receipt = positioned_paged_kv_write_compile_receipt(
    candidate,
    "d".repeat(64),
    module_,
  )
  let first = bind_positioned_paged_kv_write_compiled_artifact(
    candidate,
    module_,
    module_,
    receipt,
    paged_kv_write_test_limits(),
  )
  let second = bind_positioned_paged_kv_write_compiled_artifact(
    candidate,
    module_,
    module_,
    receipt,
    paged_kv_write_test_limits(),
  )
  assert_eq(first.binding_digest(), second.binding_digest())
  assert_false(first.manifest_bindable())
  assert_false(first.promotion_authority())
  try
    bind_positioned_paged_kv_write_compiled_artifact(
      candidate,
      module_,
      b"substituted",
      receipt,
      paged_kv_write_test_limits(),
    )
  catch {
    Mismatch(Determinism) => ()
    _ => fail("unexpected compiled replay rejection")
  } noraise {
    _ => fail("non-deterministic module reached compiled binding")
  }
  let replay_receipt = positioned_paged_kv_write_compile_receipt(
    candidate,
    "e".repeat(64),
    b"different-cubin",
  )
  try
    bind_positioned_paged_kv_write_compiled_artifact(
      candidate,
      module_,
      module_,
      replay_receipt,
      paged_kv_write_test_limits(),
    )
  catch {
    Mismatch(CompileReceipt) => ()
    _ => fail("unexpected receipt replay rejection")
  } noraise {
    _ => fail("foreign receipt reached compiled binding")
  }
}

///|
test "positioned paged KV write limits reject unbounded admission knobs" {
  for selector in 0..<5 {
    try
      ignore(
        PagedKvWriteAotLimits::new(
          max_source_bytes=if selector == 0 { 1048577 } else { 1048576 },
          max_recipe_bytes=if selector == 1 { 1048577 } else { 1048576 },
          max_module_bytes=if selector == 2 { 1073741825L } else { 1073741824L },
          max_receipt_bytes=if selector == 3 { 1048577 } else { 1048576 },
          max_binding_bytes=if selector == 4 { 1048577 } else { 1048576 },
        ),
      )
    catch {
      Invalid(Limits) => ()
      _ => fail("unexpected positioned KV-write limit rejection")
    } noraise {
      _ => fail("unbounded positioned KV-write limit was admitted")
    }
  }
}

///|
test "fake campaign rejects a silent guard return and preserves sentinels" {
  let candidate = paged_kv_write_test_candidate(paged_kv_write_test_fixture())
  let observations : Array[PagedKvWriteFakeCampaignObservation] = []
  for tokens in 1..<=4 {
    observations.push(
      PagedKvWriteFakeCampaignObservation::new(
        live_tokens=tokens,
        kv_width=64,
        key_writes=tokens.to_int64() * 64L,
        value_writes=tokens.to_int64() * 64L,
        expected_untouched=31L,
        observed_untouched=31L,
      ),
    )
  }
  let report = evaluate_positioned_paged_kv_write_fake_campaign(
    candidate,
    observations[:],
  )
  assert_eq(report.observation_count(), 4)
  assert_false(report.physical_evidence())
  let silent = observations.copy()
  silent[0] = PagedKvWriteFakeCampaignObservation::new(
    live_tokens=1,
    kv_width=64,
    key_writes=0L,
    value_writes=0L,
    expected_untouched=31L,
    observed_untouched=31L,
  )
  try
    evaluate_positioned_paged_kv_write_fake_campaign(candidate, silent[:])
  catch {
    Mismatch(Fallback) => ()
    _ => fail("unexpected fake guard-return rejection")
  } noraise {
    _ => fail("silent guard return passed fake campaign")
  }
}
