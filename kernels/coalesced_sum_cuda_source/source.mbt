///|
pub fn source(plan : @parallel.CoalescedSumPlan, symbol~ : String) -> String {
  let (rows, left, right) = plan.geometry()
  // Integer bit copies preserve signed zero/NaN payloads. A complete kernel
  // boundary publishes the concatenation before the existing all-reduce.
  let kernels = [false, true].map(unpack => {
    let name = symbol + (if unpack { "_unpack" } else { "_pack" })
    let transfer = if unpack {
      "if(i<a) left[i]=joined[i];else right[i-a]=joined[i];"
    } else {
      "joined[i]=i<a?left[i]:right[i-a];"
    }
    (
      $|extern "C" __global__ void \{name}(const int *counts,unsigned *left,unsigned *right,unsigned *joined) {
      $|  int rows=counts[3];if(rows<=0||rows>\{rows})return;
      $|  unsigned long long i=(unsigned long long)blockIdx.x*256+threadIdx.x;
      $|  unsigned long long a=(unsigned long long)rows*\{left},n=(unsigned long long)rows*\{left+right};
      $|  if(i<n) {\{transfer}}
      $|}
    )
  })
  kernels.join("\n")
}
