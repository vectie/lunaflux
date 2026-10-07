///|
/// Terminal realization of one immutable published-tile read plan. Independent
/// right reads overlap instruction latency; products remain ascending. No
/// global/shared allocation, new synchronization or numerical law is added.
fn cuda_key_operand_read_pipeline(
  binding : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
) -> String {
  let plan = binding.key_operand_reads().unwrap()
  let (columns, slots) = plan.geometry()
  let ownership = binding.key_read_ownership().unwrap()
  // This CUDA instruction maps four eight-provider groups to four packed
  // words. The abstract permutation is consumed here, never in model/IR code.
  guard [
      for group in 0..<4 => ownership.provider(group * 8).unwrap() / 8
    ] ==
    [0, 2, 1, 3] &&
    slots == 2 &&
    columns >= slots &&
    columns <= 8 else {
    abort("unsupported key read ownership realization")
  }
  let outputs = columns * 8
  let address = outputs + 4
  let step = address + 1
  let source = StringBuilder()
  source <+
    "static __device__ __forceinline__ void lf_key_operand_pipeline(LfMatrixAccumulator (&score)[LF_KEY_VALUE_TILE/16], const LfMatrixOperand &a, const __nv_bfloat16 *p, int reduction) {\n"
  source <+ "  const int lane=threadIdx.x%32;\n"
  source <+
    "  const unsigned origin=(unsigned)__cvta_generic_to_shared(p+lf_operand_index((lane&7)+lane/16*8,reduction+(lane/8%2)*8,LF_HEAD_DIMENSION));\n"
  source <+
    "  asm volatile(\n    \"{ .reg .b32 b<\{slots * 4}>; .reg .u32 address; mov.u32 address, %\{address};\"\n"
  for action in plan.actions() {
    match action {
      ReadOperand(item~, slot~) => {
        if item > 0 {
          source <+ "    \"add.u32 address, address, %\{step};\"\n"
        }
        let first = slot * 4
        source <+
          "    \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {b\{first},b\{first+1},b\{first+2},b\{first+3}}, [address];\"\n"
      }
      ConsumeOperand(item~, slot~) =>
        for half in 0..<2 {
          let c = item * 8 + half * 4
          let b = slot * 4 + half * 2
          source <+
            "    \"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%\{c},%\{c+1},%\{c+2},%\{c+3}}, {%\{outputs},%\{outputs+1},%\{outputs+2},%\{outputs+3}}, {b\{b},b\{b+1}}, {%\{c},%\{c+1},%\{c+2},%\{c+3}};\"\n"
        }
    }
  }
  source <+ "    \"}\"\n    : "
  let results : Array[String] = []
  for column in 0..<columns {
    for item in 0..<8 {
      results.push("\"+f\"(score[\{column}].x[\{item}])")
    }
  }
  source.write_string(results.join(","))
  source <+
    "\n    : \"r\"(a.x[0]),\"r\"(a.x[1]),\"r\"(a.x[2]),\"r\"(a.x[3]),\"r\"(origin),\"n\"(16*LF_HEAD_DIMENSION*2) : \"memory\");\n}\n"
  source.to_string()
}
