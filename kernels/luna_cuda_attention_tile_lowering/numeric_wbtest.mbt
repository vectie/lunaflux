///|
test "CUDA terminal does not silently turn noncausal IR into causal attention" {
  let (program, optimization, schedule) = cuda_lower_test_program_and_schedule(
    causal=false,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=false,
  )
  try
    lower_attention_tile_schedule_to_cuda(
      program, optimization, schedule, target, 262144L,
    )
  catch {
    InvalidCudaAttentionTileLowering(UnsupportedCompute) => ()
    _ => fail("unexpected unsupported numeric contract result")
  } noraise {
    _ => fail("noncausal program was silently lowered")
  }
}
