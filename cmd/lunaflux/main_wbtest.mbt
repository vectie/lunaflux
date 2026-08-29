///|
test "reference arguments are bounded before artifact loading" {
  assert_eq(parse_positive("4"), 4)
  assert_eq(parse_token_ids("1,42,99"), [1, 42, 99])
  for text in ["0", "4097", "invalid"] {
    try parse_positive(text) catch {
      InvalidReferenceArgument(MaximumNewTokens) => ()
      _ => fail("unexpected maximum-new-token error")
    } noraise {
      _ => fail("expected invalid maximum new tokens")
    }
  }
  for text in ["", "1,", "-1", "x"] {
    try parse_token_ids(text) catch {
      InvalidReferenceArgument(TokenIds) => ()
      _ => fail("unexpected token ID error")
    } noraise {
      _ => fail("expected invalid token IDs")
    }
  }
}

///|
test "operator commands cannot label a preflight as ready work" {
  assert_eq(
    operator_command_header("run"),
    "LunaFlux legacy compatibility model run preflight",
  )
  assert_eq(
    operator_command_header("doctor"),
    "LunaFlux legacy compatibility physical diagnostic",
  )
  assert_eq(operator_command_terminal_line("run"), "service started: false")
  assert_eq(
    operator_command_terminal_line("doctor"),
    "diagnostic report available: false",
  )
  assert_eq(
    operator_command_terminal_line("plan"),
    "resolved plan available: false",
  )
  assert_eq(operator_command_terminal_line("bench"), "benchmark started: false")
  assert_eq(
    operator_command_terminal_line("inspect-kernels"),
    "kernel admission available: false",
  )
  assert_eq(
    pinned_terminal_line("inspect-kernels", true),
    "kernel admission available: true",
  )
  assert_eq(pinned_terminal_line("plan", true), "resolved plan available: true")
  assert_eq(
    pinned_terminal_line("doctor", true),
    "diagnostic report available: true",
  )
  assert_eq(pinned_terminal_line("run", true), "service started: false")
  assert_eq(pinned_terminal_line("bench", true), "benchmark started: false")
  assert_eq(
    runtime_report_rejection_line(DeviceBytesOverflow),
    "runtime report rejected: device byte total overflow",
  )
}

///|
test "native cleanup retries a failed first drain request" {
  assert_eq(native_drain_action(Ready, false), NativeDrainRequest)
  // An ownership rejection may leave the phase unchanged until the next turn.
  assert_eq(native_drain_action(Ready, false), NativeDrainRequest)
  // The production owner latches Failed while retaining the server; progress
  // then owns the lower begin-drain retry and cannot report completion early.
  assert_eq(native_drain_action(Failed, false), NativeDrainProgress)
  assert_eq(native_drain_action(Failed, true), NativeDrainComplete)
}

///|
test "opaque run CLI admits exactly one deployment argument" {
  assert_eq(
    exact_opaque_run_argument(["lunaflux", "run", "/deployment#sha256=digest"]),
    Some("/deployment#sha256=digest"),
  )
  for
    hostile in [
      ["lunaflux", "run"],
      ["lunaflux", "run", "deployment", "fallback"],
      ["lunaflux", "doctor", "deployment"],
    ] {
    assert_true(exact_opaque_run_argument(hostile) is None)
  }
}

///|
test "opaque bench CLI admits exactly one deployment argument" {
  assert_eq(
    exact_opaque_bench_argument([
      "lunaflux", "bench", "/deployment#sha256=digest",
    ]),
    Some("/deployment#sha256=digest"),
  )
  for
    hostile in [
      ["lunaflux", "bench"],
      ["lunaflux", "bench", "deployment", "fallback"],
      ["lunaflux", "run", "deployment"],
    ] {
    assert_true(exact_opaque_bench_argument(hostile) is None)
  }
}

///|
test "canonical diagnostics admit exactly one opaque deployment argument" {
  let deployment = "/deployment#sha256=" + "a".repeat(64)
  assert_eq(
    exact_release_diagnostic_argument(["lunaflux", "doctor", deployment]),
    Some(ReleaseDoctor(deployment)),
  )
  assert_eq(
    exact_release_diagnostic_argument(["lunaflux", "plan", deployment]),
    Some(ReleasePlan(deployment)),
  )
  assert_eq(
    exact_release_diagnostic_argument([
      "lunaflux", "inspect-kernels", deployment,
    ]),
    Some(ReleaseInspectKernels(deployment)),
  )
  for
    hostile in [
      ["lunaflux", "doctor"],
      ["lunaflux", "doctor", deployment, "fallback"],
      ["lunaflux", "legacy-doctor", deployment],
      ["lunaflux", "run", deployment],
      ["lunaflux", "plan", deployment, "fallback"],
      ["lunaflux", "inspect-kernels"],
    ] {
    assert_true(exact_release_diagnostic_argument(hostile) is None)
  }
}

///|
test "canonical diagnostics publish only inert root-free terminal claims" {
  let deployment = "/deployment#sha256=" + "a".repeat(64)
  let commands = [
    ReleaseDoctor(deployment),
    ReleasePlan(deployment),
    ReleaseInspectKernels(deployment),
  ]
  for command in commands {
    let header = release_diagnostic_header(command)
    let terminal = release_diagnostic_terminal_line(command)
    assert_true(header.contains("authenticated deployment"))
    assert_true(terminal.has_suffix("available: true"))
    assert_false(header.contains("/deployment"))
    assert_false(terminal.contains("/deployment"))
    assert_false(terminal.contains("readiness: true"))
  }
}

///|
test "only visibly named commands enter legacy compatibility" {
  assert_eq(legacy_pinned_command("legacy-run"), Some("run"))
  assert_eq(legacy_pinned_command("legacy-doctor"), Some("doctor"))
  assert_eq(legacy_pinned_command("legacy-plan"), Some("plan"))
  assert_eq(legacy_pinned_command("legacy-bench"), Some("bench"))
  assert_eq(
    legacy_pinned_command("legacy-inspect-kernels"),
    Some("inspect-kernels"),
  )
  for bare in ["run", "doctor", "plan", "bench", "inspect-kernels"] {
    assert_true(legacy_pinned_command(bare) is None)
  }
}

///|
test "opaque CLI startup measurement is exact and payload-safe" {
  let (latency, ready) = startup_measurement_lines(321, RuntimeReady)
  assert_eq(latency, "cold_start_latency_millis: 321")
  assert_eq(ready, "startup_measurement_readiness: true")
  let (maximum, not_ready) = startup_measurement_lines(
    86400000,
    RuntimeNotReady,
  )
  assert_eq(maximum, "cold_start_latency_millis: 86400000")
  assert_eq(not_ready, "startup_measurement_readiness: false")
  for line in [latency, ready, maximum, not_ready] {
    assert_false(line.contains("model"))
    assert_false(line.contains("path"))
    assert_false(line.contains("secret"))
  }
}

///|
test "opaque CLI publishes exact machine-readable serving and control origins" {
  assert_eq(
    runtime_origin_line("http://127.0.0.1:8080"),
    "runtime_origin=http://127.0.0.1:8080",
  )
  assert_eq(
    control_origin_line("http://127.0.0.1:43127"),
    "control_origin=http://127.0.0.1:43127",
  )
}
