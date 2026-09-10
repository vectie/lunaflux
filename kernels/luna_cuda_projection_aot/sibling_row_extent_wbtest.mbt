///|
fn sibling_bounded_source_for_test(
  maximum_tokens : Int,
  groups : Int,
) -> String raise {
  let parameters = sibling_resource_test_parameters()
  let target = projection_fixture_target()
  let baseline = projection_build_runtime_dispatch(parameters, 1024, target, [])
  let dispatch_rows = Int::max(8, maximum_tokens)
  let strategy = sibling_resource_test_strategy(baseline, dispatch_rows, groups)
  let runtime = projection_build_runtime_dispatch(parameters, 1024, target, [
    row_variant_test_record(parameters, strategy),
  ])
  let dispatch = runtime.dispatch_plan(dispatch_rows)
  let compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
      dispatch~,
      maximum_rows=Int::min(32, maximum_tokens),
      maximum_tokens~,
      demand=AllTokenRows,
      single_row_numerics=AllowDeterministicDotTree,
    ),
  )
  let combined = @utf8.decode(
    projection_source(
      "bounded_gate",
      parameters,
      maximum_tokens,
      Int::min(32, maximum_tokens),
      dispatch,
      compilation,
      false,
      true,
      max_registers=128,
    ),
  )
  combined.split("extern \"C\" __global__ void bounded_gate_down(").to_array()[0].to_owned()
}

///|
test "bounded sibling rows remove dead staging and accumulator products" {
  for maximum_tokens in [2, 8, 16] {
    for groups in [1, 2, 4, 8, 16] {
      let source = sibling_bounded_source_for_test(maximum_tokens, groups)
      let columns = Int::max(1, groups / 4) * 16
      let width = if groups <= 8 { 64 } else { 32 }
      let slot = (16 + 2 * columns) * width
      assert_true(source.contains("sibling_storage[\{slot * 2}]"))
      assert_true(source.contains("((token_count + 15) / 16)"))
      assert_true(source.contains("float cg[1][8] = {}, cu[1][8] = {};"))
      assert_false(source.contains("float cg[2]"))
      assert_false(source.contains("float cg[4]"))
      assert_false(source.contains("__launch_bounds__"))
      // Unused row owners disappear before CUDA rendering. Live tail lanes
      // remain masked; the ordered K16 fold and storage permutation persist.
      assert_true(source.contains("inner < 1024; inner += \{width}"))
      assert_true(source.contains("k < \{width}; k += 16"))
      assert_true(slot * 4 + groups * 1024 <= 49152)
      assert_true(source.contains("if (output_row < token_count)"))
      assert_true(source.contains("operand_index(row, k)"))
      assert_true(source.contains("mma.sync.aligned.m16n8k16"))
      assert_true(source.contains("if (token_count == 1)"))
    }
  }
}

///|
test "bounded sibling staging fences remain outside consumer and tail guards" {
  let source = sibling_bounded_source_for_test(8, 2)
  assert_true(source.contains("if (warp < 1)"))
  // Initial stage completion and every slot rotation publish to the entire
  // workgroup. No consumer predicate or masked row can bypass a CTA fence.
  assert_true(
    source.contains(
      "stage_operands(0, 0);\n      asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");\n      __syncthreads();",
    ),
  )
  assert_true(
    source.contains(
      "        }\n        asm volatile(\"cp.async.wait_group 0;\" ::: \"memory\");\n        __syncthreads();\n        stage ^= 1;",
    ),
  )
  assert_true(
    source.contains(
      "if (row_begin + (consumer_row + row) * 16 >= token_count) break;",
    ),
  )
  assert_true(
    source.contains("const bool valid = row_start + row < token_count;"),
  )
  assert_false(source.contains("is_input"))
  assert_false(source.contains("is_gate"))
  assert_true(source.contains("cp.async.cg.shared.global"))
  assert_true(source.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"))
}

///|
test "larger sibling envelopes retain bounded reuse and the measured resource topology" {
  for row_case in [(17, 32), (32, 32), (33, 48), (49, 64), (1024, 64)] {
    let (maximum_tokens, rows) = row_case
    let source = sibling_bounded_source_for_test(maximum_tokens, 16)
    assert_true(source.contains("((token_count + \{rows - 1}) / \{rows})"))
    if rows == 64 {
      assert_true(source.contains("__launch_bounds__(512, 2)"))
      assert_true(source.contains("float c[2][8] = {};"))
      assert_false(source.contains("cp.async"))
    } else {
      assert_false(source.contains("__launch_bounds__"))
    }
    if rows == 48 {
      // Three row microtiles need fewer compute owners. Copy ownership still
      // spans every launched thread; a 256-thread stride would duplicate the
      // first half's next vector across the second half's current vector.
      assert_true(source.contains("if (warp < 8)"))
      assert_true(
        source.contains("const int vector = threadIdx.x + slot * 512;"),
      )
      assert_false(
        source.contains("const int vector = threadIdx.x + slot * 256;"),
      )
      assert_true(source.contains("__syncthreads();\n        stage ^= 1;"))
    }
  }
}

///|
test "partial sibling consumers do not duplicate shared transfer ownership" {
  for groups in [4, 8, 16] {
    let rows = 48
    let columns = Int::max(1, groups / 4) * 16
    let threads = groups * 32
    let vectors = (rows + columns * 2) * 32 / 8
    let batches = (vectors + threads - 1) / threads
    let writes = Array::make(vectors, 0)
    for thread in 0..<threads {
      for slot in 0..<batches {
        let vector = thread + slot * threads
        if vector < vectors {
          writes[vector] += 1
        }
      }
    }
    assert_true(writes.all(fn(count) { count == 1 }))
    let source = sibling_bounded_source_for_test(33, groups)
    assert_true(
      source.contains("const int vector = threadIdx.x + slot * \{threads};"),
    )
    assert_true(source.contains("float *gate_tiles"))
    // The operand staging dies before the register-major sibling epilogue
    // aliases it. Both result planes remain within the admitted storage.
    let staged_bytes = (rows + columns * 2) * 32 * 2 * 2
    assert_true(rows * columns * 4 <= staged_bytes)
    assert_true(rows * columns * 4 <= groups * 1024)
  }
}
