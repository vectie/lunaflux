///|
test "resident counters use the implemented core clock" {
  fn rejected(f : () -> Unit raise) -> Bool {
    try {
      f()
      false
    } catch {
      _ => true
    }
  }
  assert_eq(benchmark_clocks("30000000"), (30000000, 25000000))
  assert_eq(benchmark_clocks("30000000 40000000"), (30000000, 40000000))
  assert_true(rejected(() => ignore(benchmark_clocks("30000000 123"))))
  assert_eq(benchmark_transport("PCIE 100000000 10.000"), (None, 100000000))
  assert_eq(
    benchmark_transport("30000000 100000000"),
    (Some(30000000), 100000000),
  )
  assert_true(
    rejected(() => ignore(benchmark_transport("PCIE 100000000 20.000"))),
  )
  assert_true(rejected(() => ignore(benchmark_transport("10000000 100000000"))))
  let report =
    #|CLOCK pblaze3_i/clocks/clk_core clk_core PERIOD=25.000
    #|REGISTER_CLOCK core_i 13302 clk_core
    #|REGISTER_CLOCK control_i 1746 clk_core
    #|REGISTER_CLOCK bridge_i 729 clk_core
    #|PBLAZE3_JTAG_INFERENCE_ROUTED_GUARD_COMPLETE
  assert_eq(routed_core_clock(report), 40000000)
  for
    (period, hz) in [
      ("20.000", 50000000),
      ("16.667", 60000000),
      ("13.333", 75000000),
      ("12.500", 80000000),
      ("10.000", 100000000),
    ] {
    assert_eq(routed_core_clock(report.replace(old="25.000", new=period)), hz)
    assert_eq(benchmark_clocks("30000000 " + hz.to_string()), (30000000, hz))
  }
  assert_eq(
    routed_core_clock(report.replace(old="25.000", new="40.000")),
    25000000,
  )
  assert_true(
    rejected(() => {
      ignore(
        routed_core_clock(report.replace(old="729 clk_core", new="729 clk_ui")),
      )
    }),
  )
  assert_true(rejected(() => ignore(routed_core_clock(report + "\n" + report))))
  let fractional = report.replace(old="25.000", new="14.444") +
    "\nCORE_MMCM_HZ 69230769"
  assert_eq(routed_core_clock(fractional), 69230769)
  assert_eq(benchmark_clocks("30000000 69230769 14.444"), (30000000, 69230769))
  assert_true(
    rejected(() => {
      ignore(routed_core_clock(fractional + "\nCORE_MMCM_HZ 69230769"))
    }),
  )
  assert_true(
    rejected(() => ignore(benchmark_clocks("30000000 75000000 14.444"))),
  )
}
