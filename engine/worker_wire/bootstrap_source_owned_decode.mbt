///|
// Publish validated typed records and one owned copy of the received frame.
// Decoding must not re-encode, hash, or byte-compare the entire frame merely to
// reconstruct the same owner. Callers validate their recipe and field bounds.
fn own_decoded_bootstrap_source(
  source : FixedArray[Byte],
  length : Int,
  roots~ : WorkerBootstrapRoots,
  model~ : WorkerBootstrapModelSource,
  kernel~ : WorkerBootstrapKernelSource,
  execution~ : WorkerBootstrapExecution,
  ceilings~ : WorkerBootstrapCeilings,
  recipe~ : WorkerBootstrapRecipe,
  tensor_parallel_policy? : TensorParallelBootstrapPolicy,
  i8_model_source? : WorkerI8BootstrapModelSource,
  i8_execution? : WorkerI8BootstrapExecution,
  mistral_model_source? : WorkerMistralBootstrapModelSource,
  qwen3_model_source? : WorkerQwen3BootstrapModelSource,
  fp8_model_source? : WorkerFp8BootstrapModelSource,
  fp8_execution? : WorkerFp8BootstrapExecution,
) -> EncodedBootstrapSource raise WorkerWireError {
  let digest = bootstrap_source_raw_digest(
    source,
    length - BOOTSTRAP_SOURCE_DIGEST_BYTES,
  )
  let storage = FixedArray::makei(BOOTSTRAP_SOURCE_CAPACITY, index => {
    if index < length {
      source[index]
    } else {
      b'\x00'
    }
  })
  {
    storage,
    length,
    digest,
    roots,
    model,
    kernel,
    execution,
    ceilings,
    recipe,
    tensor_parallel_policy,
    i8_model_source,
    i8_execution,
    mistral_model_source,
    qwen3_model_source,
    fp8_model_source,
    fp8_execution,
  }
}
