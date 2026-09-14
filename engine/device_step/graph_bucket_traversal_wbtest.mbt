///|
test "bucket representative traversal equals exhaustive concrete shape mapping" {
  let (table, _, _) = graph_dispatch_fixture(7)
  for rows in [1, 2, 3, 7, 8, 17, 32] {
    for tokens in [1, 2, 3, 7, 8, 17, 33, 64] {
      for mode in 0..<3 {
        let expected = FixedArray::makei(table.bucket_count(), slot => {
          if slot % 7 == 0 {
            99
          } else {
            -1
          }
        })
        let actual = expected.copy()
        for row in 1..<=rows {
          for token in 1..<=tokens {
            if mode != 0 || row <= token {
              for context = 1
                  context <= paged_graph_maximum_context_tokens()
                  context = context * 2 {
                let phase : @luna_execution_graph_strategy.ExecutionGraphPhase = if mode ==
                  0 {
                  PrefillGraph
                } else {
                  DecodeGraph
                }
                let slot = graph_dispatch_bucket(
                  table, phase, row, token, context,
                ).slot()
                if mode == 2 {
                  expected[slot] = -1
                } else if expected[slot] < 0 {
                  expected[slot] = 4
                }
                if mode == 0 && row >= 2 {
                  let mixed = graph_dispatch_bucket(
                    table,
                    MixedGraph,
                    row,
                    token,
                    context,
                  ).slot()
                  if expected[mixed] < 0 {
                    expected[mixed] = 4
                  }
                }
              }
            }
          }
        }
        match mode {
          0 => map_paged_prefill_capture_owner(table, rows, tokens, 4, actual)
          1 => map_paged_decode_capture_owner(table, rows, tokens, 4, actual)
          _ => clear_paged_decode_capture_owners(table, rows, tokens, actual)
        }
        assert_eq(actual, expected)
      }
    }
  }
}
