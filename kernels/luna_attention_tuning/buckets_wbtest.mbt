///|
test "bucket observations preserve shape winners and reject unmatched work" {
  let buckets = @graph.build_execution_graph_bucket_table(16, 4096, 8192, 65536)
  let header = "luna-attention-buckets-v1\nscope\tfrontier-device-toolchain\n"
  let base = "record\tprefill\t8\t1528\t256\t322\t91000\t5\n"
  let fast = "record\tprefill\t8\t1528\t256\t324\t77000\t5\n"
  let routes = parse_attention_bucket_routes(
    @utf8.encode(header + base + fast),
    scope="frontier-device-toolchain",
    buckets,
    baseline=322,
    [322, 324],
  )
  let shape = @graph.ExecutionGraphShape::new(
    phase=PrefillGraph,
    batch_rows=8,
    query_tokens=1528,
    context_tokens=256,
  )
  assert_eq(routes.select(shape), Some(324))
  for
    text in [
      header,
      header + fast,
      header + base + base,
      (header + base + fast).replace(old="1528\t256\t324", new="1500\t256\t324"),
      (header + base + fast).replace(old="77000\t5", new="77000\t4"),
      (header + base + fast).replace(old="324", new="999"),
      (header + base + fast).replace(
        old="frontier-device-toolchain",
        new="stale",
      ),
      (header + base + fast).replace(old="prefill", new="decode"),
    ] {
    try
      parse_attention_bucket_routes(
        @utf8.encode(text),
        scope="frontier-device-toolchain",
        buckets,
        baseline=322,
        [322, 324],
      )
    catch {
      InvalidTable => ()
    } noraise {
      _ => fail("invalid bucket observations accepted")
    }
  }
}
