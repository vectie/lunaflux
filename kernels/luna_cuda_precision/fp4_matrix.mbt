///|
/// Terminal register ABI only. Both operands contain E2M1 codes with one
/// E4M3 scale per sixteen elements. Global scales remain explicit epilogue
/// operands. Every lane participates, including inactive logical rows.
pub fn e2m1_scaled_matrix_helper(symbol~ : String) -> String {
  (
    $|__device__ __forceinline__ void \{symbol}_mma(float *d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1,unsigned sa,unsigned sb) {
    $|  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
    $|    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    $|    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb));
    $|}
  )
}
