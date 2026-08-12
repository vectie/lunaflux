///|
priv struct FakeKvResource {
  byte_count : Int64
  context_id : Int
  mut open : Bool
}

///|
fn fake_identity(seed : String) -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(seed),
    @spec.PlanDigest::from_sha256(seed),
  )
}

///|
fn fake_layout(
  identity : @spec.ModelIdentity,
) -> @device_layout.DeviceKvLayout raise {
  @device_layout.plan(
    identity,
    @plan.KvCacheGeometry::new(
      layer_count=2,
      kv_head_count=2,
      head_dimension=4,
      dtype=BFloat16,
      layout=LayerMajorSplitKeyValueV1,
    ),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=4,
      total_page_count=3,
      max_arena_bytes=3072L,
    ),
  )
}

///|
test "semantic rejection happens before the allocation callback" {
  let opens = Ref(0)
  let layout_identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let expected_identity = fake_identity(
    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  )
  try
    prepare_owned_resource(
      expected_identity,
      fake_layout(layout_identity),
      bytes => {
        opens.val += 1
        { byte_count: bytes, context_id: 7, open: true }
      },
      resource => resource.byte_count,
      (_resource, _bytes, _alignment) => (),
      _resource => (),
    )
  catch {
    InvalidLayout(Identity) => ()
    _ => fail("unexpected semantic failure")
  } noraise {
    _ => fail("expected semantic failure")
  }
  assert_eq(opens.val, 0)
}

///|
test "successful construction allocates once and close is idempotent" {
  let opens = Ref(0)
  let validations = Ref(0)
  let closes = Ref(0)
  let validated_bytes = Ref(0L)
  let validated_alignment = Ref(0L)
  let identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let (_plan, outcome) = prepare_owned_resource(
    identity,
    fake_layout(identity),
    bytes => {
      opens.val += 1
      { byte_count: bytes, context_id: 7, open: true }
    },
    resource => resource.byte_count,
    (resource, bytes, alignment) => {
      if !resource.open {
        raise Closed
      }
      validated_bytes.val = bytes
      validated_alignment.val = alignment
      validations.val += 1
    },
    resource => {
      resource.open = false
      closes.val += 1
    },
  )
  guard outcome is OwnedReady(resource) else {
    fail("unexpected retained cleanup owner")
  }
  let state = Ref(Open)
  close_retry_safe(state.val, lifecycle => state.val = lifecycle, resource, value => {
    value.open = false
    closes.val += 1
  })
  close_retry_safe(state.val, lifecycle => state.val = lifecycle, resource, _ => {
    closes.val += 1
  })
  assert_eq(opens.val, 1)
  assert_eq(validations.val, 1)
  assert_eq(validated_bytes.val, 3072L)
  assert_eq(validated_alignment.val, 256L)
  assert_eq(closes.val, 1)
  assert_true(state.val == ClosedState)
}

///|
test "validation failure closes once before returning the original cause" {
  let closes = Ref(0)
  let identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  try
    prepare_owned_resource(
      identity,
      fake_layout(identity),
      bytes => { byte_count: bytes, context_id: 7, open: true },
      resource => resource.byte_count,
      (_resource, _bytes, _alignment) => {
        raise DeviceFailed(
          AllocationValidation,
          InvalidArgument(AllocationRegionValidate),
        )
      },
      resource => {
        resource.open = false
        closes.val += 1
      },
    )
  catch {
    DeviceFailed(
      AllocationValidation,
      InvalidArgument(AllocationRegionValidate)
    ) => ()
    _ => fail("unexpected validation failure")
  } noraise {
    _ => fail("expected validation failure")
  }
  assert_eq(closes.val, 1)
}

///|
test "cleanup failure retains retry authority until close succeeds" {
  let closes = Ref(0)
  let identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let (_plan, outcome) = prepare_owned_resource(
    identity,
    fake_layout(identity),
    bytes => { byte_count: bytes, context_id: 7, open: true },
    resource => resource.byte_count,
    (_resource, _bytes, _alignment) => {
      raise DeviceFailed(
        AllocationValidation,
        InvalidArgument(AllocationRegionValidate),
      )
    },
    resource => {
      closes.val += 1
      if closes.val == 1 {
        raise DeviceFailed(AllocationClose, Busy(AllocationClose))
      }
      resource.open = false
    },
  )
  guard outcome is OwnedCleanupRequired(failed) else {
    fail("expected retained cleanup owner")
  }
  assert_eq(
    failed.cause,
    DeviceFailed(
      AllocationValidation,
      InvalidArgument(AllocationRegionValidate),
    ),
  )
  assert_eq(
    failed.cleanup_failure,
    DeviceFailed(AllocationClose, Busy(AllocationClose)),
  )
  let state = Ref(ClosingFailed)
  close_retry_safe(
    state.val,
    lifecycle => state.val = lifecycle,
    failed.resource,
    resource => {
      closes.val += 1
      resource.open = false
    },
  )
  close_retry_safe(
    state.val,
    lifecycle => state.val = lifecycle,
    failed.resource,
    _ => closes.val += 1,
  )
  assert_eq(closes.val, 2)
  assert_true(state.val == ClosedState)
}

///|
test "region guard rejects wrong context and closed allocation" {
  let identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let layout = fake_layout(identity)
  let resource : FakeKvResource = {
    byte_count: layout.arena_bytes(),
    context_id: 7,
    open: true,
  }
  let validations = Ref(0)
  try
    validate_owned_region(Open, resource, layout, 7, 8, 0, Key, 0, (
      _resource,
      _offset,
      _bytes,
      _alignment,
    ) => validations.val += 1)
  catch {
    WrongContext => ()
    _ => fail("unexpected context failure")
  } noraise {
    _ => fail("expected wrong context")
  }
  try
    validate_owned_region(ClosedState, resource, layout, 7, 7, 0, Key, 0, (
      _resource,
      _offset,
      _bytes,
      _alignment,
    ) => validations.val += 1)
  catch {
    Closed => ()
    _ => fail("unexpected closed failure")
  } noraise {
    _ => fail("expected closed allocation")
  }
  assert_eq(validations.val, 0)
}

///|
test "region guard validates the exact reserved segment" {
  let identity = fake_identity(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  )
  let layout = fake_layout(identity)
  let resource : FakeKvResource = {
    byte_count: layout.arena_bytes(),
    context_id: 7,
    open: true,
  }
  let validated_offset = Ref(-1L)
  let validated_bytes = Ref(-1L)
  let validated_alignment = Ref(-1L)
  let validated_context = Ref(-1)
  let region = validate_owned_region(
    Open,
    resource,
    layout,
    7,
    7,
    1,
    Value,
    2,
    (checked, offset, bytes, alignment) => {
      if !checked.open {
        raise Closed
      }
      validated_context.val = checked.context_id
      validated_offset.val = offset
      validated_bytes.val = bytes
      validated_alignment.val = alignment
    },
  )
  assert_eq(validated_context.val, 7)
  assert_eq(validated_offset.val, 2816L)
  assert_eq(validated_bytes.val, 256L)
  assert_eq(validated_alignment.val, 256L)
  assert_eq(region.offset_bytes(), 2816L)
  assert_eq(region.end_offset_bytes(), 3072L)
}
