///|
// Use the implemented clock of the core, instruction controller and AXI
// bridge. The report is emitted by the routed bitstream's physical clock gate.
fn routed_core_clock(report : String) -> Int32 raise {
  let mut clock = ""
  let mut period = ""
  let mut exact_hz = None
  let mut complete = false
  let registers : Map[String, String] = Map([])
  for line in report.split("\n") {
    match line.split(" ").to_array() {
      ["CLOCK", "pblaze3_i/clocks/clk_core", name, value] => {
        guard period == "" && value.strip_prefix("PERIOD=") is Some(ns) else {
          fail("invalid or duplicate core clock")
        }
        clock = name.to_owned()
        period = ns.to_owned()
      }
      ["CORE_MMCM_HZ", value] => {
        guard exact_hz is None else { fail("duplicate MMCM frequency") }
        exact_hz = Some(@string.parse_int(value))
      }
      ["CORE_MMCM_HZ", ..] => fail("malformed MMCM frequency")
      ["REGISTER_CLOCK", block, count, name] => {
        guard @string.parse_int(count) > 0 &&
          !registers.contains(block.to_owned()) else {
          fail("invalid routed register group")
        }
        registers[block.to_owned()] = name.to_owned()
      }
      ["PBLAZE3_JTAG_INFERENCE_ROUTED_GUARD_COMPLETE"] => complete = true
      _ => ()
    }
  }
  guard complete && period != "" else {
    fail("incomplete routed core clock report")
  }
  for block in ["core_i", "control_i", "bridge_i"] {
    guard registers.get(block) == Some(clock) else {
      fail("routed register clock mismatch")
    }
  }
  if exact_hz is Some(hz) {
    checked_period_hz(hz, period)
  } else {
    match period {
      "40.000" => 25000000
      "25.000" => 40000000
      "20.000" => 50000000
      "16.667" => 60000000
      "13.333" => 75000000
      "12.500" => 80000000
      "10.000" => 100000000
      _ => fail("unsupported routed core period")
    }
  }
}

///|
fn checked_period_hz(hz : Int32, ns : StringView) -> Int32 raise {
  guard hz > 0 &&
    (@string.parse_double(ns) - 1000000000.0 / hz.to_double()).abs() <= 0.00051 else {
    fail("MMCM frequency and routed period disagree")
  }
  hz
}

///|
fn benchmark_clocks(reply : String) -> (Int32, Int32) raise {
  match reply.split(" ").to_array() {
    [tck] => (@string.parse_int(tck), 25000000)
    [tck, "40000000"] => (@string.parse_int(tck), 40000000)
    [tck, "50000000"] => (@string.parse_int(tck), 50000000)
    [tck, "60000000"] => (@string.parse_int(tck), 60000000)
    [tck, "75000000"] => (@string.parse_int(tck), 75000000)
    [tck, "80000000"] => (@string.parse_int(tck), 80000000)
    [tck, "100000000"] => (@string.parse_int(tck), 100000000)
    [tck, hz, ns] =>
      (@string.parse_int(tck), checked_period_hz(@string.parse_int(hz), ns))
    _ => fail("unsupported hardware clock reply")
  }
}

///|
fn benchmark_transport(reply : String) -> (Int32?, Int32) raise {
  match reply.split(" ").to_array() {
    ["PCIE", hz, period] =>
      (None, checked_period_hz(@string.parse_int(hz), period))
    _ => {
      let (tck, hz) = benchmark_clocks(reply)
      assert_eq(tck, 30000000, msg="30 MHz benchmark requires accepted TCK")
      (Some(tck), hz)
    }
  }
}
