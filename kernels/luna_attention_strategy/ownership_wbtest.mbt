///|
test "query-owned fragments require an explicit numerical contract" {
  let base = compiler_test_problem()
  let allowed = { ..base, allow_alternative_softmax_reduction: true, }
  let capabilities = compiler_test_capabilities()
  assert_false(
    generate_attention_tile_candidates(base, capabilities).any(c => {
      c.stable_id() == 318
    }),
  )
  for id in [318, 319] {
    let candidate = compiler_candidate_with_id(
      generate_attention_tile_candidates(allowed, capabilities),
      id,
    )
    let (q, k, d) = candidate.tile().dimensions()
    let (subgroups, _, _) = candidate.execution_geometry()
    assert_eq(
      attention_fragment_ownership(
        candidate.compute(),
        query_rows=q,
        key_rows=k,
        head_columns=d,
        subgroups~,
      ),
      QueryOwnedFragments,
    )
    assert_eq(
      candidate.shared_memory_bytes(),
      (q + 2 * k).to_int64() * d.to_int64() * 2L + 16L,
    )
  }
  let non_bf16 = { ..allowed, numeric_type: AttentionFloat16, }
  assert_false(
    generate_attention_tile_candidates(non_bf16, capabilities).any(c => {
      c.stable_id() == 318
    }),
  )
}

///|
test "ownership is derived from geometry not candidate identity" {
  for q in [32, 64] {
    for k in [32, 64] {
      for d in [64, 128] {
        assert_eq(
          attention_fragment_ownership(
            AttentionMatrixQkPv,
            query_rows=q,
            key_rows=k,
            head_columns=d,
            subgroups=q / 16,
          ),
          QueryOwnedFragments,
        )
        assert_eq(
          attention_fragment_ownership(
            AttentionMatrixQkPv,
            query_rows=q,
            key_rows=k,
            head_columns=d,
            subgroups=8,
          ),
          ExchangedScoreTiles,
        )
      }
    }
  }
}

///|
test "dense current read view substitutes only the current interval" {
  for origin in [0, 1, 15, 4096] {
    for count in [0, 1, 16, 129] {
      for position in 0..<(origin + count + 2) {
        let got = attention_key_value_read_address(
          DenseCurrentAndPagedHistory,
          position~,
          current_begin=origin,
          current_count=count,
        )
        let expected = if position >= origin && position < origin + count {
          CurrentRow(position - origin)
        } else {
          HistoricalPosition(position)
        }
        assert_eq(got, Some(expected))
        assert_eq(
          attention_key_value_read_address(
            PagedCacheOnly,
            position~,
            current_begin=origin,
            current_count=count,
          ),
          Some(HistoricalPosition(position)),
        )
      }
    }
  }
  assert_eq(
    attention_key_value_read_address(
      DenseCurrentAndPagedHistory,
      position=0,
      current_begin=2147483647,
      current_count=1,
    ),
    None,
  )
  assert_eq(
    attention_key_value_read_address(
      DenseCurrentAndPagedHistory,
      position=-1,
      current_begin=0,
      current_count=1,
    ),
    None,
  )
}
