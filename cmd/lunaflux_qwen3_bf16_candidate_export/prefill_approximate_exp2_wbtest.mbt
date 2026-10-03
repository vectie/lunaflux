///|
fn approximate_cli_base() -> Array[String] {
  let digest = "a".repeat(64)
  [
    "export", "/model", "config.json", digest, digest, "weights.numeric", digest,
    digest, digest, "13", "1", "115", "8", "8192", "32", "256", "5120", "/new-output",
  ]
}

///|
test "export CLI requires explicit approximate exponential permission" {
  let raw = approximate_cli_base()
  assert_false(
    parse_qwen3_candidate_export_arguments(raw).prefill_approximate_exp2,
  )
  raw.append(["--prefill-candidate", "30322"])
  try parse_qwen3_candidate_export_arguments(raw) catch {
    InvalidArguments => ()
    _ => fail("unexpected numerical permission error")
  } noraise {
    _ => fail("approximate candidate accepted without permission")
  }
  raw.push("--prefill-approximate-exp2")
  let parsed = parse_qwen3_candidate_export_arguments(raw)
  assert_true(parsed.prefill_approximate_exp2)
  assert_eq(parsed.prefill_candidate, Some(30322))
  raw.push("--prefill-approximate-exp2")
  try parse_qwen3_candidate_export_arguments(raw) catch {
    InvalidArguments => ()
    _ => fail("unexpected duplicate numerical permission error")
  } noraise {
    _ => fail("duplicate numerical permission accepted")
  }
}

///|
test "wide and retained approximate identities cannot bypass CLI permission" {
  for option in ["--prefill-candidate", "--wide-prefill-candidate"] {
    for id in [30322, 70644, 80645] {
      let raw = approximate_cli_base()
      raw.append([option, id.to_string()])
      try parse_qwen3_candidate_export_arguments(raw) catch {
        InvalidArguments => ()
        _ => fail("unexpected retained numerical permission error")
      } noraise {
        _ => fail("retained approximate candidate accepted without permission")
      }
    }
  }
  let strict = approximate_cli_base()
  strict.append(["--prefill-candidate", "322"])
  assert_false(
    parse_qwen3_candidate_export_arguments(strict).prefill_approximate_exp2,
  )
}

///|
test "approximate export namespace cannot relabel a strict selected winner" {
  for selection in [[], ["--prefill-candidate", "322"]] {
    let raw = approximate_cli_base()
    raw.append(selection)
    raw.push("--prefill-approximate-exp2")
    try parse_qwen3_candidate_export_arguments(raw) catch {
      InvalidArguments => ()
      _ => fail("unexpected strict relabeling rejection")
    } noraise {
      _ => fail("strict selection relabeled as approximate")
    }
  }
  let mixed = approximate_cli_base()
  mixed.append([
    "--prefill-candidate", "30322", "--wide-prefill-candidate", "322", "--prefill-approximate-exp2",
  ])
  assert_eq(
    parse_qwen3_candidate_export_arguments(mixed).wide_prefill_candidate,
    Some(322),
  )
}
