///|
/// The functional transform proves one RHS product consumer in the current
/// published epoch. Keep its load and the ordered consumers inside one PTX
/// region: no returned C++ RHS aggregate, indexing temporary or copied pair.
/// PTX virtual registers still need hardware allocation; fewer SASS MOVs and
/// lower latency require counters, not a source-level assumption.
fn cuda_forwarded_matrix_products(
  binding : @luna_cuda_attention_tile_lowering.CudaPrefillFragmentBinding,
) -> String {
  let source = StringBuilder()
  let lifetime = binding.query_lifetime()
  for
    (name, plan, transpose) in [
      ("key", binding.score(), ""),
      ("value", binding.value(), ".trans"),
    ] {
    // Complete joint consumption has no remaining per-key call. Emitting an
    // unused helper fails warning-denied AOT compilation. Epoch-local joint
    // consumption also covers every reduction fragment; partial retention
    // still needs the ordinary remainder. Derive reachability from the plan.
    if name == "key" &&
      lifetime.joint_right_consumers() &&
      (
        lifetime.retained_dimensions() == 0 ||
        lifetime.retained_dimensions() == lifetime.program().fold().geometry().2
      ) {
      continue
    }
    if name == "key" &&
      binding.key_operand_reads() is Some(reads) &&
      reads.geometry().1 > 1 {
      source.write_string(cuda_key_operand_read_pipeline(binding))
      continue
    }
    let b00 = plan.word(0, 0).unwrap()
    let b01 = plan.word(0, 1).unwrap()
    let b10 = plan.word(1, 0).unwrap()
    let b11 = plan.word(1, 1).unwrap()
    let (parameters, address) = if name == "value" {
      ("unsigned address", "")
    } else {
      (
        "const __nv_bfloat16 *p, int row, int column, int columns",
        "  const int lane = threadIdx.x % 32;\n" +
        (if binding.key_read_ownership() is Some(_) {
          "  const unsigned address = (unsigned)__cvta_generic_to_shared(p + lf_operand_index(row + (lane & 7) + lane / 16 * 8, column + (lane / 8 % 2) * 8, columns));\n"
        } else {
          "  const unsigned address = (unsigned)__cvta_generic_to_shared(p + lf_operand_index(row + lane % 16, column + lane / 16 * 8, columns));\n"
        }),
      )
    }
    source.write_string(
      (
        $|static __device__ __forceinline__ void lf_matrix_mma_\{name}(LfMatrixAccumulator &c, const LfMatrixOperand &a, \{parameters}) {
        $|\{address}
        $|  asm volatile(
        $|    "{ .reg .b32 b<4>;"
        $|    "ldmatrix.sync.aligned.m8n8.x4\{transpose}.shared.b16 {b0,b1,b2,b3}, [%12];"
        $|    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%8,%9,%10,%11}, {b\{b00},b\{b01}}, {%0,%1,%2,%3};"
        $|    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%4,%5,%6,%7}, {%8,%9,%10,%11}, {b\{b10},b\{b11}}, {%4,%5,%6,%7};"
        $|    "}"
        $|    : "+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3]),"+f"(c.x[4]),"+f"(c.x[5]),"+f"(c.x[6]),"+f"(c.x[7])
        $|    : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(address) : "memory");
        $|}
      ) +
      "\n",
    )
  }
  source.to_string()
}
