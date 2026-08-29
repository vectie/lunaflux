///|
test "rank configure rejects policy ordinal substitution" {
  let source = configure_test_source()
  let startup = configure_test_startup(0, source)
  try
    compose_rank_configure(
      configure_test_binding(0, startup),
      startup,
      source,
      configure_test_topology(swapped=true),
      configure_test_envelope(0, startup),
    )
  catch {
    Invalid(Topology) => ()
    _ => fail("unexpected policy ordinal rejection")
  } noraise {
    _ => fail("policy ordinal substitution was admitted")
  }
}

///|
test "rank configure rejects collective version outside source policy" {
  let source = configure_test_source()
  let startup = configure_test_startup(0, source)
  try
    compose_rank_configure(
      configure_test_binding(0, startup),
      startup,
      source,
      configure_test_topology(),
      configure_test_envelope(0, startup, collective_runtime_version_code=22000),
    )
  catch {
    Invalid(RankEnvelope) => ()
    _ => fail("unexpected collective version rejection")
  } noraise {
    _ => fail("out-of-policy collective version was admitted")
  }
}
