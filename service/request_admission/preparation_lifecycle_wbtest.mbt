///|
test "deadline samples once before and once after tokenization" {
  let identity = fixture_identity("0")
  let reads = Ref(0)
  let receipt : RequestReceipt = { at_millis: 10UL }
  let request = fixture_request(identity)
  let prepared = prepare_bound_with_clock(
    receipt,
    fixture_deadline(receipt, request),
    request,
    fixture_tokenizer(),
    fixture_tokenizer().digest(),
    identity,
    fixture_inference(),
    () => {
      reads.val += 1
      if reads.val == 1 {
        11UL
      } else {
        12UL
      }
    },
  )
  assert_eq(reads.val, 2)
  assert_eq(prepared.input_token_count(), 3)
  let expired_reads = Ref(0)
  try
    prepare_bound_with_clock(
      receipt,
      fixture_deadline(receipt, request),
      request,
      fixture_tokenizer(),
      fixture_tokenizer().digest(),
      identity,
      fixture_inference(),
      () => {
        expired_reads.val += 1
        110UL
      },
    )
  catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected pre-tokenization deadline error")
  } noraise {
    _ => fail("expired request reached tokenization")
  }
  assert_eq(expired_reads.val, 1)
  let post_expired_reads = Ref(0)
  try
    prepare_bound_with_clock(
      receipt,
      fixture_deadline(receipt, request),
      request,
      fixture_tokenizer(),
      fixture_tokenizer().digest(),
      identity,
      fixture_inference(),
      () => {
        post_expired_reads.val += 1
        if post_expired_reads.val == 1 {
          11UL
        } else {
          110UL
        }
      },
    )
  catch {
    Invalid(Deadline) => ()
    _ => fail("unexpected post-tokenization deadline error")
  } noraise {
    _ => fail("request expiring during tokenization was prepared")
  }
  assert_eq(post_expired_reads.val, 2)
  let failed_reads = Ref(0)
  try
    prepare_bound_with_clock(
      receipt,
      fixture_deadline(receipt, request),
      request,
      fixture_tokenizer(),
      fixture_tokenizer().digest(),
      identity,
      fixture_inference(),
      () => {
        failed_reads.val += 1
        raise ClockUnavailable
      },
    )
  catch {
    ClockUnavailable => ()
    _ => fail("unexpected preparation clock error")
  } noraise {
    _ => fail("failed preparation clock published an owner")
  }
  assert_eq(failed_reads.val, 1)
}

///|
test "retained prepared aliases lose authority after destructive take" {
  let identity = fixture_identity("0")
  let prepared = prepare_at(
    { at_millis: 10UL },
    fixture_request(identity),
    fixture_tokenizer(),
    fixture_tokenizer().digest(),
    identity,
    fixture_inference(),
    11UL,
  )
  let retained = prepared
  assert_eq(retained.input_token_count(), 3)
  let claim = prepared.take_claim()
  assert_eq(claim.scheduler_request().request_id().value(), 7UL)
  try retained.take_claim() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected retained-shell retake error")
  } noraise {
    _ => fail("retained shell preserved take authority")
  }
  try retained.model_identity() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected retained-shell getter error")
  } noraise {
    _ => fail("retained shell preserved getter authority")
  }
  let destination = FixedArray::make(8, b'!')
  let output = claim.push_token_into_status(
    65,
    destination,
    destination_offset=0,
  )
  assert_true(output.is_ready())
  assert_eq(output.copied_bytes(), 1)
  assert_eq(destination[0], b'A')
  let finished = claim.finish_into_status(destination, destination_offset=1)
  assert_true(finished.is_ready())
}
