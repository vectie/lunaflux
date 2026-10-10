///|
pub fn seeded_token_block_source(
  plan : @precision.SeededTokenBlockPrecision,
  symbol_prefix~ : String,
) -> Bytes raise {
  guard !symbol_prefix.is_empty() &&
    symbol_prefix.length() <= 80 &&
    symbol_prefix[0] is ('a'..='z' | 'A'..='Z' | '_') &&
    symbol_prefix.all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '_')) else {
    raise @precision.PrecisionPlanError::IllegalPlacement
  }
  let (v, rows, main_rows, noise) = plan.geometry()
  let source =
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbol_prefix}_seed(const int32_t* main_counts,const int32_t* start,const int32_t* seed,int32_t* counts,int32_t* ids,int32_t* positions,int32_t* chain) {
    $|  if(threadIdx.x!=0) return;
    $|  const int n=main_counts[3],token=seed[0];
    $|  const bool valid=n>0 && n<=\{main_rows} && start[0]>=0 && start[0]<=2147483647-n-\{rows} && token>=0 && token<\{v};
    $|  counts[0]=counts[1]=counts[4]=0;counts[2]=1;counts[3]=valid?\{rows}:0;
    $|  chain[0]=valid?token:-1;
    $|  for(int row=0;row<\{rows};++row) {
    $|    ids[row]=row==0?token:\{noise};
    $|    positions[row]=valid?start[0]+n+row:0;
    $|  }
    $|}
  @utf8.encode(source)
}
