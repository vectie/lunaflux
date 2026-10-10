///|
async test "connected observations do not scan unselected modules or rehash routes" {
  @async.with_task_group() <| group => {
    let directory = @fs.tmpdir(prefix="lunaflux-connected-label")
    group.add_defer() <| () => {
      @async.protect_from_cancel(() => @fs.rmdir(directory, recursive=true))
    }
    let raw = reference_prefill_test_args()
    let path = directory + "/comparison"
    let routes = directory + "/routes"
    @fs.write_file(path, chain_fixture(raw, 3), create_mode=CreateNew)
    // Fixture modules deliberately do not exist: binding an observation must
    // not read every competing module before selection and compilation.
    let (rest, measured) = read_connected_choice(
      [..raw, "--prefill-chain-comparison", path, "f".repeat(64)],
    )
    assert_eq(rest.to_array(), raw)
    assert_true(measured is Some(_))
    let route_bytes = b"supplied route plan"
    @fs.write_file(routes, route_bytes, create_mode=CreateNew)
    let original = @utf8.decode(chain_fixture(raw, 3))
    let label = "\"routes_path\":\"" +
      routes +
      "\",\"routes_sha256\":\"" +
      "f".repeat(64) +
      "\","
    let with_routes = parse_prefill_chain_measurements(
      @utf8.encode(
        original.replace(
          old="\"path\":\"/tmp/kv128.cubin\"",
          new=label + "\"path\":\"/tmp/kv128.cubin\"",
        ),
      ),
    )
    assert_eq(connected_route_bytes(with_routes, 0), Some(route_bytes))
    assert_eq(connected_route_bytes(with_routes, 2), None)
  }
}

///|
fn chain_fixture(raw : ArrayView[String], winner : Int) -> Bytes {
  let value : Json = {
    "schema": "lunaflux-prefill-connected-comparison.v1",
    "context_sha256": chain_context_sha256(raw, raw[71]),
    "numeric_law": "reference-bf16-probability-exp2-v1",
    "measurement_scope": "test fixture, not physical evidence",
    "vendor_rows": "2048",
    "modules": [
      { "path": "/tmp/kv128.cubin", "sha256": "a".repeat(64) },
      { "path": "/tmp/kv64.cubin", "sha256": "b".repeat(64) },
    ],
    "observations": [
      for id in 0..<4 => {
        {
          "plan": id.to_string(),
          "nanoseconds": if id == winner {
            "100"
          } else {
            "200"
          },
          "samples": "3",
        }
      }
    ],
  }
  @utf8.encode(value.stringify())
}

///|
test "complete continuation binds both choices and both attention slots" {
  let raw = reference_prefill_test_args()
  let (args, ingress, law, _) = prepare_prefill_numerical_choice(raw)
  for winner in 0..<4 {
    let measured = parse_prefill_chain_measurements(chain_fixture(raw, winner))
    let (bound, vendor, selected) = bind_connected_choice(
      raw, args, ingress, law, measured,
    )
    assert_eq(selected, winner)
    assert_eq(vendor, if winner % 2 == 1 { 2048 } else { 0 })
    let attention = winner / 2
    assert_eq(
      bound.attention_prefill.module_path,
      measured.modules[attention].0,
    )
    assert_eq(
      bound.attention_prefill_wide.module_path,
      bound.attention_prefill.module_path,
    )
    assert_eq(
      bound.attention_prefill.shared_memory_bytes,
      if attention == 0 {
        81920
      } else {
        49152
      },
    )
    assert_eq(
      bound.attention_prefill_wide.shared_memory_bytes,
      bound.attention_prefill.shared_memory_bytes,
    )
    // Changing the destination is safe, changing a fixed compiler choice is not.
    let changed = raw.copy()
    changed[71] = "/tmp/new-output.bundle"
    assert_eq(
      chain_context_sha256(changed, changed[71]),
      measured.context_sha256,
    )
    changed[18] = "128"
    assert_not_eq(
      chain_context_sha256(changed, changed[71]),
      measured.context_sha256,
    )
    try bind_connected_choice(changed, args, ingress, law, measured) catch {
      InvalidArguments => ()
      e => raise e
    } noraise {
      _ => fail("stale comparison accepted")
    }
  }
}

///|
test "complete continuation rejects implicit numeric law and incomplete observations" {
  let raw = reference_prefill_test_args()
  let (args, ingress, _, _) = prepare_prefill_numerical_choice(raw)
  let bytes = chain_fixture(raw, 3)
  let measured = parse_prefill_chain_measurements(bytes)
  try
    bind_connected_choice(
      raw,
      args,
      ingress,
      StrictNaturalExponential,
      measured,
    )
  catch {
    InvalidArguments => ()
    e => raise e
  } noraise {
    _ => fail("numeric contract changed silently")
  }
  for
    (before, after) in [
      ("\"samples\":\"3\"", "\"samples\":\"2\""),
      ("\"plan\":\"2\"", "\"plan\":\"1\""),
      ("\"vendor_rows\":\"2048\"", "\"vendor_rows\":\"1\""),
    ] {
    let bad = @utf8.encode(@utf8.decode(bytes).replace(old=before, new=after))
    try parse_prefill_chain_measurements(bad) catch {
      InvalidArguments => ()
      e => raise e
    } noraise {
      _ => fail("incomplete or inconsistent measurements accepted")
    }
  }
}

///|
test "connected alternatives retain paired route identities without relabeling" {
  let original = @utf8.decode(chain_fixture(reference_prefill_test_args(), 3))
  let absent = parse_prefill_chain_measurements(@utf8.encode(original))
  assert_eq(absent.route_files, [None, None])
  let entry = "\"path\":\"/tmp/kv128.cubin\""
  let routes = "\"routes_path\":\"/tmp/kv128.routes\",\"routes_sha256\":\"" +
    "c".repeat(64) +
    "\","
  let valid = parse_prefill_chain_measurements(
    @utf8.encode(original.replace(old=entry, new=routes + entry)),
  )
  assert_eq(valid.route_files, [
    Some(("/tmp/kv128.routes", "c".repeat(64))),
    None,
  ])
  for
    fragment in [
      "\"routes_path\":\"/tmp/kv128.routes\",",
      "\"routes_sha256\":\"" + "c".repeat(64) + "\",",
      routes.replace(old="/tmp/kv128.routes", new="relative.routes"),
      routes.replace(old="c".repeat(64), new="invalid"),
    ] {
    try
      parse_prefill_chain_measurements(
        @utf8.encode(original.replace(old=entry, new=fragment + entry)),
      )
    catch {
      InvalidArguments => ()
      e => raise e
    } noraise {
      _ => fail("incomplete route identity accepted")
    }
  }
}
