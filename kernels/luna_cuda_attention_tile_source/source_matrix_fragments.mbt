///|
/// CUDA-only register ownership for an ordered BF16 matrix fold. Both shared
/// layouts are permutations; no extra storage or numerical reassociation.
fn cuda_attention_matrix_fragments() -> String {
  (
    #|static __device__ __forceinline__ int lf_float_index(int row, int column, int columns) {
    #|  const unsigned r = (unsigned)row & 7, c = (unsigned)column & 15;
    #|  const unsigned bank = (r * 4) ^ ((r & 1) * 8) ^ (c >> 1) ^ ((c & 1) * 16) ^ ((((unsigned)column >> 4) & 1) * 8);
    #|  return (((unsigned)row >> 3) * (columns / 16) + ((unsigned)column >> 4)) * 128 + ((c >> 3) + (c & 1) * 2) * 32 + bank;
    #|}
    #|struct LfMatrixOperand { unsigned x[4]; };
    #|struct LfMatrixAccumulator { float x[8]; static constexpr int num_elements = 8; };
    #|static __device__ __forceinline__ int lf_matrix_row(int item, int lane) { return lane / 4 + (item % 4 / 2) * 8; }
    #|static __device__ __forceinline__ int lf_matrix_column(int item, int lane) { return lane % 4 * 2 + item % 2 + item / 4 * 8; }
    #|static __device__ __forceinline__ LfMatrixOperand lf_matrix_load(const __nv_bfloat16 *p, int row, int column, int columns, bool transpose = false) {
    #|  const int lane = threadIdx.x % 32;
    #|  const unsigned address = (unsigned)__cvta_generic_to_shared(p + lf_operand_index(row + lane % 16, column + lane / 16 * 8, columns));
    #|  LfMatrixOperand a;
    #|  if (transpose) {
    #|    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]) : "r"(address));
    #|  } else {
    #|    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a.x[0]),"=r"(a.x[1]),"=r"(a.x[2]),"=r"(a.x[3]) : "r"(address));
    #|  }
    #|  return a;
    #|}
    #|static __device__ __forceinline__ void lf_matrix_mma(LfMatrixAccumulator &c, const LfMatrixOperand &a, const LfMatrixOperand &b, bool row_b = false) {
    #|  #pragma unroll
    #|  for (int half = 0; half < 2; ++half) {
    #|    float *d = c.x + half * 4;
    #|    const unsigned b0 = b.x[row_b ? half * 2 : half], b1 = b.x[row_b ? half * 2 + 1 : half + 2];
    #|    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a.x[0]),"r"(a.x[1]),"r"(a.x[2]),"r"(a.x[3]),"r"(b0),"r"(b1));
    #|  }
    #|}
    #|static __device__ __forceinline__ void lf_matrix_store_float(float *p, int row, int column, int columns, const LfMatrixAccumulator &c) {
    #|  #pragma unroll
    #|  for (int item = 0; item < 8; ++item) p[lf_float_index(row + lf_matrix_row(item, threadIdx.x % 32), column + lf_matrix_column(item, threadIdx.x % 32), columns)] = c.x[item];
    #|}
  )
}

///|
/// Only a shared-resident PV fold reloads its F32 accumulator. A register-
/// resident fold must not emit an unused helper for a materialization it elides.
fn cuda_attention_matrix_float_load() -> String {
  (
    #|static __device__ __forceinline__ LfMatrixAccumulator lf_matrix_load_float(const float *p, int row, int column, int columns) {
    #|  LfMatrixAccumulator c;
    #|  #pragma unroll
    #|  for (int item = 0; item < 8; ++item) c.x[item] = p[lf_float_index(row + lf_matrix_row(item, threadIdx.x % 32), column + lf_matrix_column(item, threadIdx.x % 32), columns)];
    #|  return c;
    #|}
  )
}
