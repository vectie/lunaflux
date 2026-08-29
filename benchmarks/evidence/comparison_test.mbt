///|
test "complete counterbalanced comparison produces stable canonical evidence" {
  let first = @evidence.BenchmarkComparison::admit(
    environment=test_environment(),
    engines=test_engines(),
    workloads=test_workloads(),
    trials=test_trials(true),
  )
  let second = @evidence.BenchmarkComparison::admit(
    environment=test_environment(),
    engines=test_engines(),
    workloads=test_workloads(),
    trials=test_trials(true),
  )
  assert_eq(first.trial_count(), 81)
  assert_eq(first.digest(), second.digest())
  assert_eq(first.canonical_bytes(), second.canonical_bytes())
}

///|
test "outcome accounting and correctness fail before performance admission" {
  try
    @evidence.BenchmarkTrial::new(
      engine=LunaFlux,
      profile=Latency,
      ordinal=1,
      order_position=0,
      submitted=2,
      completed=1,
      rejected=0,
      timed_out=0,
      cancelled=0,
      failed=0,
      correctness_passed=true,
      raw_events=test_digest('1'),
      correctness=test_digest('2'),
      summary=test_digest('3'),
    )
  catch {
    @evidence.IncorrectOutcomeAccounting => ()
    _ => fail("unexpected outcome-accounting error")
  } noraise {
    _ => fail("unaccounted request was admitted")
  }
  try
    @evidence.BenchmarkTrial::new(
      engine=LunaFlux,
      profile=Latency,
      ordinal=1,
      order_position=0,
      submitted=1,
      completed=1,
      rejected=0,
      timed_out=0,
      cancelled=0,
      failed=0,
      correctness_passed=false,
      raw_events=test_digest('1'),
      correctness=test_digest('2'),
      summary=test_digest('3'),
    )
  catch {
    @evidence.CorrectnessNotEstablished => ()
    _ => fail("unexpected correctness error")
  } noraise {
    _ => fail("incorrect result was admitted")
  }
}

///|
test "missing profiles and biased engine order fail closed" {
  let workloads = test_workloads()
  ignore(workloads.pop())
  try
    @evidence.BenchmarkComparison::admit(
      environment=test_environment(),
      engines=test_engines(),
      workloads~,
      trials=test_trials(true),
    )
  catch {
    @evidence.InvalidWorkload => ()
    _ => fail("unexpected workload error")
  } noraise {
    _ => fail("incomplete workload envelope was admitted")
  }
  try
    @evidence.BenchmarkComparison::admit(
      environment=test_environment(),
      engines=test_engines(),
      workloads=test_workloads(),
      trials=test_trials(false),
    )
  catch {
    @evidence.BiasedEngineOrder => ()
    _ => fail("unexpected engine-order error")
  } noraise {
    _ => fail("engine order was not counterbalanced")
  }
}

///|
test "engine identities and exact trial matrix cannot be cherry-picked" {
  let engines = test_engines()
  engines[1] = @evidence.BenchmarkEngineIdentity::new(
    engine=Sglang,
    revision=test_digest('b'),
    executable=test_digest('c'),
  )
  try
    @evidence.BenchmarkComparison::admit(
      environment=test_environment(),
      engines~,
      workloads=test_workloads(),
      trials=test_trials(true),
    )
  catch {
    @evidence.InvalidEngineIdentity => ()
    _ => fail("unexpected identity error")
  } noraise {
    _ => fail("duplicate engine identity was admitted")
  }
  let trials = test_trials(true)
  ignore(trials.pop())
  try
    @evidence.BenchmarkComparison::admit(
      environment=test_environment(),
      engines=test_engines(),
      workloads=test_workloads(),
      trials~,
    )
  catch {
    @evidence.InvalidComparisonShape => ()
    _ => fail("unexpected trial-shape error")
  } noraise {
    _ => fail("incomplete trial matrix was admitted")
  }
  let duplicate = test_trials(true)
  duplicate[1] = duplicate[0]
  try
    @evidence.BenchmarkComparison::admit(
      environment=test_environment(),
      engines=test_engines(),
      workloads=test_workloads(),
      trials=duplicate,
    )
  catch {
    @evidence.DuplicateTrial => ()
    _ => fail("unexpected duplicate-trial error")
  } noraise {
    _ => fail("duplicate trial was admitted")
  }
}

///|
test "artifact digests accept only canonical lowercase sha256" {
  for
    value in [
      "", "ABC", "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
    ] {
    try @evidence.BenchmarkDigest::from_sha256(value) catch {
      InvalidDigest => ()
      _ => fail("unexpected digest error")
    } noraise {
      _ => fail("invalid digest was admitted")
    }
  }
}
