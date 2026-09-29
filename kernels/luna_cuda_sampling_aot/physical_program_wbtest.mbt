///|
test "sampling physical migration preserves emitted sources" {
  let greedy = embedded_greedy_sampling_cuda_source(151936, 32)
  inspect(
    sampling_sha256(@utf8.encode(greedy)),
    content=(
      #|559a56695e296877fb70133231cfdad1845423f628f212fc74b8d78d90dfca3e
    ),
  )
}

///|
test "stochastic physical migration preserves source" {
  let plan = @luna_sampling_strategy.plan_device_sampling(
    maximum_rows=4,
    vocabulary_size=151936,
    maximum_top_k=64,
    maximum_workspace_bytes=3072L,
  )
  inspect(
    sampling_sha256(
      @utf8.encode(
        device_sampling_source_text(plan, stochastic_physical_program(plan)),
      ),
    ),
    content=(
      #|18c45f119c2b99c2197416b5a00014760b406e9c34afe3c682cba3bee0d54bcb
    ),
  )
}
