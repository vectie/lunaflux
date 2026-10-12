///|
fn projection_source(
  plan : @precision.MoeRoutingPrecision,
  symbol : String,
  weight_type : String,
  weight_load : String,
) -> (String, (Int, Int)) raise {
  let (rows, hidden, experts, _) = plan.geometry()
  let grid = (rows.to_int64() * experts.to_int64() + 255L) / 256L
  match plan.projection_law() {
    OrderedFloat32 => {
      let source =
        $|extern "C" __global__ void \{symbol}_project(const int *counts, const __nv_bfloat16 *input, const \{weight_type} *weight, float *logits) {
        $|  const int live=counts[3]; if(live<=0||live>\{rows}) return;
        $|  const unsigned long long cell=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        $|  if(cell>=(unsigned long long)live*\{experts}ULL) return;
        $|  const unsigned long long row=cell/\{experts}ULL, base=(cell%\{experts}ULL)*\{hidden}ULL;
        $|  float sum=0.0f; for(int k=0;k<\{hidden};++k) sum=__fadd_rn(sum,__fmul_rn(__bfloat162float(input[row*\{hidden}ULL+k]),\{weight_load}));
        $|  logits[cell]=sum;
        $|}
      (source, (grid.to_int(), 256))
    }
    GroupedVectorFloat32(
      groups=1,
      partitions~,
      vector_width=1,
      publication_partitions=1,
      multiply_add=FusedMultiplyAdd
    ) if partitions <= 1024 => {
      let cells = rows.to_int64() * experts.to_int64()
      guard cells <= 2147483647L else {
        raise @precision.PrecisionPlanError::ArithmeticOverflow
      }
      let source =
        $|extern "C" __global__ void \{symbol}_project(const int *counts, const __nv_bfloat16 *input, const \{weight_type} *weight, float *logits) {
        $|  const int live=counts[3]; if(live<=0||live>\{rows}) return;
        $|  const unsigned long long cell=blockIdx.x, row=cell/\{experts}ULL, base=(cell%\{experts}ULL)*\{hidden}ULL;
        $|  if(row>=(unsigned long long)live) return;
        $|  const int lane=threadIdx.x; __shared__ float partial[\{partitions}]; float sum=0.0f;
        $|  for(int k=lane;k<\{hidden};k+=\{partitions}) sum=__fmaf_rn(__bfloat162float(input[row*\{hidden}ULL+k]),\{weight_load},sum);
        $|  partial[lane]=sum; __syncthreads();
        $|  for(int span=\{partitions / 2};span>0;span>>=1) { if(lane<span) partial[lane]=__fadd_rn(partial[lane],partial[lane+span]); __syncthreads(); }
        $|  if(lane==0) logits[cell]=partial[0];
        $|}
      (source, (cells.to_int(), partitions))
    }
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
}
