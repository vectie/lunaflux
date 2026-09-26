///|
fn family_streaming_claims(
  family : Int,
  text : String,
) -> StreamingClaims? raise {
  match family {
    0 => parse_i8_fixture(text).streaming
    1 =>
      parse_fp8_descriptor(
        @utf8.encode(text),
        RuntimeDescriptorFileLimits::new(
          max_descriptor_bytes=16384L,
          max_json_depth=16,
        ),
      ).streaming
    2 => parse_qwen3_fixture(text).streaming
    3 => parse_mistral_fixture(text).streaming
    _ => parse_tensor_parallel_fixture(text).streaming
  }
}

///|
test "each worker family requires explicit versioned streaming activation" {
  let fp8 = i8_descriptor_fixture()
    .replace(
      old="lunaflux.runtime.i8.v2",
      new="lunaflux.runtime.fp8-reusable.v3",
    )
    .replace(old=I8_RUNTIME_POLICY, new=FP8_RUNTIME_POLICY)
    .replace(
      old=",\"admitted_bootstrap_v3_sha256\":\"4123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"",
      new="",
    )
  let streaming =
    #|"streaming":{"host_slots":4,"host_budget_bytes":1024,"reserve_pages":4,"restore_nanoseconds_per_page":100,"recompute_nanoseconds_per_page":1000,"separate_host_memory":true,"transfer_timeout_millis":10000,"workspace_budget_bytes":1024,"prefill_shapes":[{"tokens":1,"workspace_bytes":128},{"tokens":4,"workspace_bytes":1024}]},
  let fixtures = [
    i8_descriptor_fixture(),
    fp8,
    qwen3_descriptor_fixture(),
    mistral_descriptor_fixture(),
    tensor_parallel_descriptor_fixture(),
  ]
  let versions = [
    ("lunaflux.runtime.i8.v2", "lunaflux.runtime.i8.v3"),
    ("lunaflux.runtime.fp8-reusable.v3", "lunaflux.runtime.fp8-reusable.v4"),
    ("lunaflux.runtime.qwen3_bf16.v1", "lunaflux.runtime.qwen3_bf16.v3"),
    ("lunaflux.runtime.mistral_bf16.v1", "lunaflux.runtime.mistral_bf16.v3"),
    (
      "lunaflux.runtime.tensor-parallel.v1", "lunaflux.runtime.tensor-parallel.v2",
    ),
  ]
  for family, fixture in fixtures {
    assert_true(family_streaming_claims(family, fixture) is None)
    let extended = fixture.replace(
      old="\"model\":",
      new=streaming + "\"model\":",
    )
    try family_streaming_claims(family, extended) catch {
      _ => ()
    } noraise {
      _ => fail("old schema accepted streaming")
    }
    let (old, next) = versions[family]
    let versioned = extended.replace(old~, new=next)
    let text = if family == 3 || family == 4 {
      versioned
    } else {
      versioned.replace(
        old="\"max_total_module_bytes\":4096",
        new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":16384",
      )
    }
    let claims = family_streaming_claims(family, text).unwrap()
    assert_eq(claims.slots, 4)
    assert_eq(claims.shapes.length(), 2)
    for
      invalid in [
        fixture.replace(old~, new=next),
        text.replace(old="\"transfer_timeout_millis\":10000,", new=""),
        text.replace(old="\"tokens\":1", new="\"tokens\":0"),
      ] {
      try family_streaming_claims(family, invalid) catch {
        _ => ()
      } noraise {
        _ => fail("incomplete streaming policy accepted")
      }
    }
  }
}
