///|
/// Consume one Q representation in a single ordered fan-out region. The
/// lifetime plan chooses an invariant binding or an epoch-local binding.
/// The row XOR repeats every eight rows, so sixteen-row instruction strides
/// preserve it. One K scratch fragment is consumed before reuse. No extra K
/// live state or per-accumulator numerical reassociation is introduced.
fn cuda_joint_query_consumer(
  binding : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
  query_fragment : Int,
) -> String {
  let plan = binding.score()
  let (_, columns, _, quantum, _) = plan.program().geometry()
  let (_, _, dimensions) = binding.program().fold().geometry()
  let row_step = binding.program().geometry().0
  guard row_step == 16 && quantum == 16 && columns > 0 && columns <= 8 else {
    abort("unsupported joint query consumer instruction geometry")
  }
  let operand_base = columns * 8
  let address_operand = operand_base + 4
  let query_operand = if binding.query_lifetime().retained_fragments() == 0 {
    "query_operand"
  } else {
    "query_operand_\{query_fragment}"
  }
  let b00 = plan.word(0, 0).unwrap()
  let b01 = plan.word(0, 1).unwrap()
  let b10 = plan.word(1, 0).unwrap()
  let b11 = plan.word(1, 1).unwrap()
  let source = StringBuilder()
  source <+
    "        // joint-right-columns-v1: one immutable Q input binding.\n"
  source <+ "        {\n"
  source <+ "          const int query_lane=threadIdx.x%32;\n"
  source <+
    "          const unsigned key_base=(unsigned)__cvta_generic_to_shared(current_k+lf_operand_index(query_lane%16,\{query_fragment * quantum}+query_lane/16*8,LF_HEAD_DIMENSION));\n"
  source <+ "          asm volatile(\n"
  source <+ "            \"{ .reg .b32 b<4>; .reg .u32 key_address;\"\n"
  source <+ "            \"mov.u32 key_address, %\{address_operand};\"\n"
  for key in 0..<columns {
    if key > 0 {
      source <+
        "            \"add.u32 key_address, key_address, \{row_step * dimensions * 2};\"\n"
    }
    source <+
      "            \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {b0,b1,b2,b3}, [key_address];\"\n"
    for half in 0..<2 {
      let output = key * 8 + half * 4
      let (b0, b1) = if half == 0 { (b00, b01) } else { (b10, b11) }
      source <+
        "            \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%\{output},%\{output+1},%\{output+2},%\{output+3}}, {%\{operand_base},%\{operand_base+1},%\{operand_base+2},%\{operand_base+3}}, {b\{b0},b\{b1}}, {%\{output},%\{output+1},%\{output+2},%\{output+3}};\"\n"
    }
  }
  source <+ "            \"}\"\n"
  let outputs : Array[String] = []
  for key in 0..<columns {
    for item in 0..<8 {
      outputs.push("\"+f\"(score[\{key}].x[\{item}])")
    }
  }
  source <+ "            : \{outputs.join(",")}\n"
  let inputs = [ for item in 0..<4 => "\"r\"(\{query_operand}.x[\{item}])" ]
  source <+ "            : \{inputs.join(",")},\"r\"(key_base) : \"memory\");\n"
  source <+ "        }\n"
  source.to_string()
}
