///|
test "v6 explicitly binds bounded streaming policy while legacy schema rejects it" {
  let streaming =
    #|"streaming":{"host_slots":4,"host_budget_bytes":256,"reserve_pages":4,"restore_nanoseconds_per_page":100,"recompute_nanoseconds_per_page":1000,"separate_host_memory":true,"transfer_timeout_millis":10000,"workspace_budget_bytes":1024,"prefill_shapes":[{"tokens":1,"workspace_bytes":128},{"tokens":4,"workspace_bytes":1024}]},
  let text = graph_descriptor_fixture()
    .replace(old="lunaflux.runtime.v1", new="lunaflux.runtime.v6")
    .replace(old="\"model\":", new=streaming + "\"model\":")
    .replace(
      old="\"max_total_module_bytes\":4096",
      new="\"max_total_module_bytes\":4096,\"max_graph_capture_bytes\":8192",
    )
  let claims = parse_graph_descriptor_fixture(text)
  assert_eq(claims.streaming.unwrap().shapes.length(), 2)
  assert_eq(claims.streaming.unwrap().workspace_budget, 1024L)
  for
    invalid in [
      text.replace(old="lunaflux.runtime.v6", new="lunaflux.runtime.v2"),
      text.replace(old="\"tokens\":1", new="\"tokens\":0"),
      text.replace(old="\"workspace_budget_bytes\":1024,", new=""),
    ] {
    try parse_graph_descriptor_fixture(invalid) catch {
      _ => ()
    } noraise {
      _ => fail("invalid streaming descriptor accepted")
    }
  }
}
