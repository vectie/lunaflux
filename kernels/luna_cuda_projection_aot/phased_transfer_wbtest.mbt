///|
test "sibling register transfer keeps thread local reads and late addresses" {
  let source = projection_sibling_phased_transfer_source(64, 64, 32, 1024, 512)
  assert_true(source.contains("unsigned values[2][4] = {};"))
  let phases = source.split("const unsigned address").collect()
  assert_eq(phases.length(), 2)
  assert_true(phases[0].contains("ld.global.v4.u32"))
  assert_false(phases[0].contains("st.shared.v4.u32"))
  assert_false(phases[1].contains("ld.global"))
  assert_true(phases[1].contains("st.shared.v4.u32"))
  assert_false(source.contains("addresses["))
  assert_false(source.contains("__syncthreads"))
  assert_false(source.contains("cp.async"))
  assert_true(source.contains("if (valid) {"))
}

///|
test "sibling consumption precedes next stage writes and uniform publication" {
  let parameters : ProjectionParameters = {
    family: GatedMlp,
    input_width: 1024,
    output_width: 1024,
    query_width: 0,
    intermediate_width: 3072,
  }
  let runtime = projection_build_runtime_dispatch(
    parameters,
    1024,
    projection_fixture_target(),
    [],
  )
  let source = row_variant_primary_test_source(
    parameters,
    runtime.dispatch_plan(1024),
  )
  let functions = source
    .split("extern \"C\" __global__ void baseline_down(")
    .collect()
  assert_eq(functions.length(), 2)
  let gate = functions[0]
  assert_false(gate.contains("cp.async"))
  assert_eq(gate.split("__syncthreads();").count(), 5)
  assert_true(gate.contains("stage_operands(0, 0);\n      __syncthreads();"))
  let loop_parts = gate
    .split("if (inner + 32 < 1024) stage_operands(inner + 32, stage ^ 1);")
    .collect()
  assert_eq(loop_parts.length(), 2)
  assert_true(loop_parts[0].contains("mma.sync.aligned"))
  assert_true(
    loop_parts[1].has_prefix("\n        __syncthreads();\n        stage ^= 1;"),
  )
  // The final iteration still crosses this fence before operand storage
  // changes ownership to the gate epilogue; no tail-dependent fence exists.
  assert_true(gate.contains("stage ^= 1;\n      }\n      float *gate_tiles"))
}

///|
test "phased transfer batching covers arbitrary tile products once" {
  for rows in [16, 32, 64, 128] {
    for columns in [16, 32, 64, 128] {
      for width in [16, 32, 64, 128] {
        for threads in [32, 64, 128, 256, 512] {
          let vectors = (rows + columns * 2) * width / 8
          let batches = (vectors + threads - 1) / threads
          let seen = FixedArray::make(vectors, false)
          for batch in 0..<batches {
            for lane in 0..<threads {
              let vector = lane + batch * threads
              if vector < vectors {
                assert_false(seen[vector])
                seen[vector] = true
              }
            }
          }
          for present in seen {
            assert_true(present)
          }
          let source = projection_sibling_phased_transfer_source(
            rows,
            columns,
            width,
            width * 8,
            threads,
          )
          assert_true(source.contains("unsigned values[\{batches}][4]"))
          assert_true(source.contains("vector < \{vectors}"))
        }
      }
    }
  }
}
