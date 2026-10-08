///|
/// Optional terminal implementation of the same read-only paged operation.
/// The model adapter supplies an authenticated shape, never a model-name
/// heuristic. Emission is not selection: the named numerical law needs an
/// explicit bundle choice and independent accuracy qualification.
fn reference_prefill_plan(
  source : @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi,
) -> @luna_cuda_reference_attention.ReferenceAttentionPlan? {
  let (query_heads, key_value_heads, head_dimension, tokens_per_page) = source.shape()
  let (input_row_width, page_stride_values, physical_page_capacity) = source.layout()
  let (maximum_query_rows, maximum_query_tokens, maximum_page_entries) = source.limits()
  let abi = @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi::new(
    query_heads~,
    key_value_heads~,
    head_dimension~,
    tokens_per_page~,
    input_row_width~,
    page_stride_values~,
    physical_page_capacity~,
    maximum_query_rows~,
    maximum_query_tokens~,
    maximum_page_entries~,
  ) catch {
    _ => return None
  }
  Some(
    @luna_cuda_reference_attention.ReferenceAttentionPlan::new(
      backend=FlashAttentionMixed,
      abi~,
    ),
  ) catch {
    _ => None
  }
}

///|
async fn publish_reference_prefill_candidate(
  output : String,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  operation : Int,
  abi : @luna_cuda_attention_tile_source.CudaPagedAttentionTileAbi,
) -> Unit raise Qwen3Bf16CandidateExportCliError {
  guard reference_prefill_plan(abi) is Some(reference) else {
    println("reference_prefill=unsupported-shape")
    return
  }
  let (gx, gy, _, bx, _, _, shared) = reference.launch()
  let compiler = set.compiler_policy()
  let identity = compiler.identity()
  let root = output + "/reusable-reference-prefill-attention"
  let recipe = @utf8.encode(
    "schema=lunaflux-reference-attention-cuda-aot-candidate.v1\n" +
    "family=paged-attention-prefill-reference\n" +
    "model_operation_id=\{operation}\nfunction_symbol=\{reference.symbol()}\n" +
    "target=sm_\{set.target().compute_major()}\{set.target().compute_minor()}\n" +
    "toolchain_sha256=\{identity.toolchain_digest_sha256()}\n" +
    "compiler_version=\{identity.major()}.\{identity.minor()}.\{identity.patch()}\n" +
    "source_sha256=\{reference.source_sha256()}\n" +
    "numeric_law=\{reference.numeric_law()}\nquery_tile_rows=64\n" +
    "query_metadata_version=0\npage_table=exact-csr-bounded-tail-v1\n" +
    "dependency=flashattention-pinned-inference-csr-v1\n" +
    "dependency_kernel_sha256=242cbc331bd09d6ccfd023aae2d31461484b2a4801dcf66733b4151f9018622a\n" +
    "optimization=3\nfmad=true\nreassociate=false\nmax_registers=0" +
    qwen_compiler_single_launch_record(gx, gy, bx, shared.to_int64()) +
    "\n",
  )
  write_new_candidate_pair(root, reference.source(), recipe)
  println("reference_prefill_source=\{root}/kernel.cu")
}
