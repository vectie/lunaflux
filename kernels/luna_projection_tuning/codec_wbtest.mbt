///|
fn fixture_scope(
  device? : String = "device-a",
  toolchain? : String = "a".repeat(64),
) -> ProjectionTuningScope raise {
  ProjectionTuningScope::new(
    backend=@strategy.ProjectionBackendId::new("test-backend"),
    architecture=@strategy.ProjectionTargetId::new("test-architecture"),
    device_identity=device,
    toolchain_sha256=toolchain,
  )
}

///|
fn fixture_record(
  id? : Int = 3008,
  latency? : Int64 = 12000L,
) -> @strategy.ProjectionAutotuneRecord raise {
  let scope = fixture_scope()
  @strategy.ProjectionAutotuneRecord::new(
    key=@strategy.ProjectionAutotuneKey::new(
      backend=scope.backend,
      target=scope.architecture,
      numeric_type=ProjectionBFloat16,
      shape=@strategy.ProjectionShape::new(
        LanguageModelHead,
        input_width=1024,
        output_width=151936,
      ),
      strategy=@strategy.ProjectionStrategy::new(
        stable_id=id,
        phase=Prefill,
        kernel_class=MatrixTile,
        row_bucket=8,
        tile=@strategy.ProjectionTile::new(rows=16, columns=16, reduction=16),
        input_residency=SelectedRowTileResidency,
      ),
    ),
    median_latency_ns=latency,
    sample_count=3,
  )
}

///|
test "persisted strategies round trip every field and canonicalize record order" {
  let first = fixture_record()
  let second = fixture_record(id=3004, latency=11000L)
  let scope = fixture_scope()
  let a = ProjectionTuningSnapshot::new(scope, [first, second])
  let b = ProjectionTuningSnapshot::new(scope, [second, first])
  assert_eq(a.canonical_bytes(), b.canonical_bytes())
  let parsed = parse_projection_tuning_snapshot(a.canonical_bytes())
  assert_eq(parsed.canonical_bytes(), a.canonical_bytes())
  assert_eq(parsed.records_for(scope)[:].to_owned(), [second, first])
  assert_eq(
    parsed.records_for(scope)[0].strategy().input_residency(),
    SelectedRowTileResidency,
  )
}

///|
test "device architecture backend and toolchain scope cannot replay" {
  let snapshot = ProjectionTuningSnapshot::new(fixture_scope(), [
    fixture_record(),
  ])
  for
    expected in [
      fixture_scope(device="device-b"),
      fixture_scope(toolchain="b".repeat(64)),
      ProjectionTuningScope::new(
        backend=@strategy.ProjectionBackendId::new("other"),
        architecture=@strategy.ProjectionTargetId::new("test-architecture"),
        device_identity="device-a",
        toolchain_sha256="a".repeat(64),
      ),
      ProjectionTuningScope::new(
        backend=@strategy.ProjectionBackendId::new("test-backend"),
        architecture=@strategy.ProjectionTargetId::new("other"),
        device_identity="device-a",
        toolchain_sha256="a".repeat(64),
      ),
    ] {
    try snapshot.records_for(expected) catch {
      ScopeMismatch => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("foreign tuning scope accepted")
    }
  }
}

///|
test "duplicate and conflicting stable schedule keys are rejected" {
  for
    records in [
      [fixture_record(), fixture_record()],
      [fixture_record(), fixture_record(latency=42L)],
      [],
    ] {
    try ProjectionTuningSnapshot::new(fixture_scope(), records) catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("ambiguous tuning records accepted")
    }
  }
}

///|
test "snapshot parser rejects noncanonical truncated and overflowing records" {
  let text = @utf8.decode(
    ProjectionTuningSnapshot::new(fixture_scope(), [fixture_record()]).canonical_bytes(),
  )
  for
    invalid in [
      text + "\n",
      text.replace(old="v1", new="v2"),
      text.replace(old="12000", new="012000"),
      text.replace(old="12000", new="9223372036854775808"),
      text.replace(old="12000", new="-1"),
      text.replace(old="\t3\n", new="\t2\n"),
      text.replace(old="\t3008\t8\t", new="\t3008\t7\t"),
      text.replace(old="\t1\t1024\t", new="\t9\t1024\t"),
    ] {
    try parse_projection_tuning_snapshot(@utf8.encode(invalid)) catch {
      InvalidSnapshot => ()
      _ => fail("unexpected error")
    } noraise {
      _ => fail("malformed tuning snapshot accepted")
    }
  }
}
