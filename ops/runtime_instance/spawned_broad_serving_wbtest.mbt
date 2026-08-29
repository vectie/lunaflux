///|
fn broad_serving_wb_identity(foreign : Bool) -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      if foreign {
        "2123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      } else {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      },
    ),
    @spec.PlanDigest::from_sha256(
      if foreign {
        "3123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      } else {
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
      },
    ),
  )
}

///|
fn broad_serving_wb_frame(
  request_id : UInt64,
  maximum_new_tokens : Int,
  foreign : Bool,
) -> FixedArray[Byte] raise {
  let limits = spawned_serving_wb_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=broad_serving_wb_identity(foreign),
    input=@inference.Input::from_token_ids([1], limits),
    max_new_tokens=maximum_new_tokens,
    context_ceiling=8,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(request_id),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"broad", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
  let encoded = @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=1024),
  ).encode(request)
  let result = FixedArray::make(encoded.length(), b'\x00')
  ignore(encoded.copy_to(result, destination_offset=0))
  result
}

///|
fn broad_serving_wb_contract(
  first : FixedArray[Byte],
) -> RuntimeSpawnedBroadServingValidationContract raise {
  RuntimeSpawnedBroadServingValidationContract::new(
    first,
    broad_serving_wb_frame(2UL, 1, false),
    broad_serving_wb_frame(3UL, 2, false),
    broad_serving_wb_frame(4UL, 2, false),
    broad_serving_wb_frame(5UL, 1, true),
    broad_serving_wb_identity(false),
    1031,
    2185,
    1,
    8,
    8,
    spawned_serving_wb_limits(),
    1024,
    1024,
    5000,
  )
}

///|
fn broad_serving_wb_accepted(
  contract : RuntimeSpawnedBroadServingValidationContract,
  maximum_new_tokens : Int,
) -> @inference.StreamEvent raise {
  Accepted(
    @inference.AcceptedEvent::new(
      model_identity=contract.expected_model,
      effective_limits=@inference.EffectiveLimits::new(
        max_new_tokens=maximum_new_tokens,
        context_ceiling=contract.context_ceiling,
        spawned_serving_wb_limits(),
      ),
    ),
  )
}

///|
fn broad_serving_wb_token(
  token_id : Int,
  position : Int,
) -> @inference.StreamEvent raise {
  Token(
    @inference.TokenEvent::new(
      token_id~,
      decoded_delta=None,
      position~,
      spawned_serving_wb_limits(),
    ),
  )
}

///|
fn broad_serving_wb_usage(output_tokens : Int) -> @inference.Usage raise {
  @inference.Usage::initial(
    input_tokens=1,
    cached_input_tokens=0,
    spawned_serving_wb_limits(),
  ).advance(output_tokens~, spawned_serving_wb_limits())
}

///|
fn broad_serving_wb_completed(
  output_tokens : Int,
  reason : @inference.FinishReason,
) -> @inference.StreamEvent raise {
  Completed(
    @inference.CompletedEvent::new(
      finish_reason=reason,
      final_usage=broad_serving_wb_usage(output_tokens),
    ),
  )
}

///|
fn broad_serving_wb_expect_rule(
  run : () -> Unit raise,
  expected : RuntimeSpawnedBroadServingValidationRule,
) -> Bool {
  try run() catch {
    RuntimeSpawnedBroadServingValidationFailed(actual) => actual == expected
    _ => false
  } noraise {
    _ => false
  }
}

///|
test "broad serving contract copies every authenticated request frame" {
  let first = broad_serving_wb_frame(1UL, 1, false)
  let original = first[0]
  let contract = broad_serving_wb_contract(first)
  first[0] = first[0] ^ b'\x01'
  assert_eq(contract.first_frame[0], original)
  assert_eq(contract.second_frame[0], original)
  assert_eq(contract.saturation_tail_frame[0], original)
  assert_eq(contract.recovery_frame[0], original)
  assert_eq(contract.foreign_frame[0], original)
}

///|
test "broad serving contract rejects a non-foreign rejection probe" {
  let first = broad_serving_wb_frame(1UL, 1, false)
  try
    RuntimeSpawnedBroadServingValidationContract::new(
      first,
      broad_serving_wb_frame(2UL, 1, false),
      broad_serving_wb_frame(3UL, 2, false),
      broad_serving_wb_frame(4UL, 2, false),
      broad_serving_wb_frame(5UL, 1, false),
      broad_serving_wb_identity(false),
      1031,
      2185,
      1,
      8,
      8,
      spawned_serving_wb_limits(),
      1024,
      1024,
      5000,
    )
  catch {
    RuntimeSpawnedBroadServingValidationFailed(BroadServingContract) => ()
    _ => abort("broad contract failed at the wrong boundary")
  } noraise {
    _ => abort("same-model rejection probe was admitted")
  }
}

///|
test "broad serving event checkpoint codes are stable and payload free" {
  let rules : Array[(RuntimeSpawnedBroadServingValidationRule, String)] = [
    (BroadServingPairAccepted0, "events_pair_accepted_0"),
    (BroadServingPairAccepted1, "events_pair_accepted_1"),
    (BroadServingPairRemainingRead, "events_pair_remaining_read"),
    (
      BroadServingPairRemainingAcceptedShape,
      "events_pair_remaining_accepted_shape",
    ),
    (
      BroadServingPairRemainingToken0Shape,
      "events_pair_remaining_token_0_shape",
    ),
    (
      BroadServingPairRemainingToken1Shape,
      "events_pair_remaining_token_1_shape",
    ),
    (
      BroadServingPairRemainingTokenPosition,
      "events_pair_remaining_token_position",
    ),
    (BroadServingPairRemainingUsageShape, "events_pair_remaining_usage_shape"),
    (
      BroadServingPairRemainingCompletedShape,
      "events_pair_remaining_completed_shape",
    ),
    (
      BroadServingPairRemainingUnexpectedEvent,
      "events_pair_remaining_unexpected_event",
    ),
    (BroadServingPairRemainingCount, "events_pair_remaining_count"),
    (BroadServingRecoveryAccepted, "events_recovery_accepted"),
    (BroadServingRecoveryToken0, "events_recovery_token_0"),
    (BroadServingRecoveryToken1, "events_recovery_token_1"),
    (BroadServingRecoveryUsage, "events_recovery_usage"),
    (BroadServingRecoveryCompleted, "events_recovery_completed"),
  ]
  for entry in rules {
    let (rule, expected) = entry
    assert_eq(rule.code(), expected)
  }
}

///|
test "broad serving remaining ledger accepts only the exact event multiset" {
  let contract = broad_serving_wb_contract(
    broad_serving_wb_frame(1UL, 1, false),
  )
  let ledger = BroadServingEventLedger::new()
  let exact = [
    broad_serving_wb_accepted(contract, 2),
    broad_serving_wb_token(1031, 0),
    broad_serving_wb_token(1031, 0),
    broad_serving_wb_token(2185, 1),
    Usage(broad_serving_wb_usage(0)),
    Usage(broad_serving_wb_usage(1)),
    Usage(broad_serving_wb_usage(2)),
    broad_serving_wb_completed(0, Cancelled),
    broad_serving_wb_completed(1, TokenLimit),
    broad_serving_wb_completed(2, TokenLimit),
  ]
  for event in exact {
    ledger.record(event, contract)
  }
  assert_true(ledger.is_exact())
  ledger.require_exact()

  let incomplete = BroadServingEventLedger::new()
  incomplete.record(broad_serving_wb_accepted(contract, 2), contract)
  assert_false(incomplete.is_exact())
  assert_true(
    broad_serving_wb_expect_rule(
      () => incomplete.require_exact(),
      BroadServingPairRemainingCount,
    ),
  )

  let hostile = BroadServingEventLedger::new()
  assert_true(
    broad_serving_wb_expect_rule(
      () => hostile.record(broad_serving_wb_token(42, 0), contract),
      BroadServingPairRemainingToken0Shape,
    ),
  )
}

///|
test "broad serving remaining ledger classifies hostile event shapes" {
  let contract = broad_serving_wb_contract(
    broad_serving_wb_frame(1UL, 1, false),
  )
  let hostile_accepted : @inference.StreamEvent = Accepted(
    @inference.AcceptedEvent::new(
      model_identity=broad_serving_wb_identity(true),
      effective_limits=@inference.EffectiveLimits::new(
        max_new_tokens=2,
        context_ceiling=8,
        spawned_serving_wb_limits(),
      ),
    ),
  )
  let hostile_failed : @inference.StreamEvent = Failed(
    @inference.FailedEvent::new(
      code=@inference.PublicErrorCode::from_ascii(b"fixture"),
      retryable=false,
    ),
  )
  let cases : Array[
    (@inference.StreamEvent, RuntimeSpawnedBroadServingValidationRule),
  ] = [
    (hostile_accepted, BroadServingPairRemainingAcceptedShape),
    (broad_serving_wb_token(42, 0), BroadServingPairRemainingToken0Shape),
    (broad_serving_wb_token(42, 1), BroadServingPairRemainingToken1Shape),
    (broad_serving_wb_token(42, 2), BroadServingPairRemainingTokenPosition),
    (Usage(broad_serving_wb_usage(3)), BroadServingPairRemainingUsageShape),
    (
      broad_serving_wb_completed(2, Cancelled),
      BroadServingPairRemainingCompletedShape,
    ),
    (hostile_failed, BroadServingPairRemainingUnexpectedEvent),
  ]
  for entry in cases {
    let (event, expected) = entry
    let ledger = BroadServingEventLedger::new()
    assert_true(
      broad_serving_wb_expect_rule(
        () => ledger.record(event, contract),
        expected,
      ),
    )
  }
}

///|
test "broad serving event validators preserve checkpoint-specific failures" {
  let contract = broad_serving_wb_contract(
    broad_serving_wb_frame(1UL, 1, false),
  )
  require_broad_accepted(
    broad_serving_wb_accepted(contract, 1),
    contract,
    1,
    BroadServingPairAccepted0,
  )
  require_broad_token(
    broad_serving_wb_token(1031, 0),
    1031,
    0,
    BroadServingRecoveryToken0,
  )
  require_broad_usage(
    Usage(broad_serving_wb_usage(2)),
    1,
    2,
    BroadServingRecoveryUsage,
  )
  require_broad_completed(
    broad_serving_wb_completed(2, TokenLimit),
    1,
    2,
    TokenLimit,
    BroadServingRecoveryCompleted,
  )

  assert_true(
    broad_serving_wb_expect_rule(
      () => {
        require_broad_accepted(
          broad_serving_wb_accepted(contract, 2),
          contract,
          1,
          BroadServingPairAccepted1,
        )
      },
      BroadServingPairAccepted1,
    ),
  )
  assert_true(
    broad_serving_wb_expect_rule(
      () => {
        require_broad_token(
          broad_serving_wb_token(2185, 1),
          1031,
          0,
          BroadServingRecoveryToken0,
        )
      },
      BroadServingRecoveryToken0,
    ),
  )
  assert_true(
    broad_serving_wb_expect_rule(
      () => {
        require_broad_usage(
          Usage(broad_serving_wb_usage(1)),
          1,
          2,
          BroadServingRecoveryUsage,
        )
      },
      BroadServingRecoveryUsage,
    ),
  )
  assert_true(
    broad_serving_wb_expect_rule(
      () => {
        require_broad_completed(
          broad_serving_wb_completed(2, Cancelled),
          1,
          2,
          TokenLimit,
          BroadServingRecoveryCompleted,
        )
      },
      BroadServingRecoveryCompleted,
    ),
  )
}
