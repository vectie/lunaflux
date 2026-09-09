///|
test "phased sibling transfer keeps complete read and publish phases" {
  let source = projection_sibling_phased_transfer_source(64, 64, 32, 1024, 512)
  assert_true(source.contains("unsigned values[2][4] = {}, addresses[2] = {};"))
  let phases = source.split("__syncthreads();").collect()
  assert_eq(phases.length(), 3)
  assert_true(phases[0].contains("ld.global.v4.u32"))
  assert_false(phases[0].contains("st.shared.v4.u32"))
  assert_false(phases[1].contains("ld.global"))
  assert_true(phases[1].contains("st.shared.v4.u32"))
  assert_true(phases[2].contains("cp.async.commit_group"))
  assert_true(source.contains("if (valid) {"))
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
