///|
/// Materialize the invariant fragment prefix once, while each published query
/// epoch is live. Backend scalarization keeps each fragment independently
/// named; no runtime index or aggregate-register copy is required.
fn cuda_query_operand_bootstrap(
  binding : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
) -> String {
  let lifetime = binding.query_lifetime()
  let quantum = lifetime.program().geometry().2
  let source = StringBuilder()
  for index in 0..<lifetime.retained_fragments() {
    source <+ "    LfMatrixOperand query_operand_\{index};\n"
  }
  if lifetime.retained_fragments() > 0 {
    source <+ "    if (qr<active && key_begin<key_end) {\n"
    for index in 0..<lifetime.retained_fragments() {
      source <+
        "      query_operand_\{index}=lf_matrix_load(qbuf,qr,\{index * quantum},LF_HEAD_DIMENSION);\n"
    }
    source <+ "    }\n"
  }
  source.to_string()
}

///|
/// Preserve ascending reduction order exactly. Retained and epoch-local
/// operands have the same BF16 instruction representation and ownership.
fn cuda_retained_query_score_product(
  binding : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
) -> String {
  let lifetime = binding.query_lifetime()
  let (_, columns, _, quantum, _) = binding.score().program().geometry()
  let source = StringBuilder()
  for index in 0..<lifetime.retained_fragments() {
    if lifetime.joint_right_consumers() {
      source.write_string(cuda_joint_query_consumer(binding, index))
      continue
    }
    source <+ "        #pragma unroll\n"
    source <+ "        for (int key=0;key<\{columns};++key) {\n"
    source <+
      "          lf_matrix_mma_key(score[key],query_operand_\{index},current_k,key*16,\{index * quantum},LF_HEAD_DIMENSION);\n"
    source <+ "        }\n"
  }
  if lifetime.retained_dimensions() < lifetime.program().fold().geometry().2 {
    source <+ "        #pragma unroll\n"
    source <+
      "        for (int reduction=\{lifetime.retained_dimensions()};reduction<LF_HEAD_DIMENSION;reduction+=\{quantum}) {\n"
    source <+
      "          const auto a=lf_matrix_load(qbuf,qr,reduction,LF_HEAD_DIMENSION);\n"
    source <+ "          #pragma unroll\n"
    source <+ "          for (int key=0;key<\{columns};++key) {\n"
    source <+
      "            lf_matrix_mma_key(score[key],a,current_k,key*16,reduction,LF_HEAD_DIMENSION);\n"
    source <+ "          }\n"
    source <+ "        }\n"
  }
  source.to_string()
}
