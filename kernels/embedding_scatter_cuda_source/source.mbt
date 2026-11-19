///|
pub struct EmbeddingScatterSource {
  priv plan : @effects.EmbeddingScatterProgram
  priv symbol : String
  priv bytes : Bytes
}

///|
/// Six arguments: step counts, causal positions, feature map, live feature
/// count, BF16 feature rows, and the BF16 residual. Decode/text rows are skipped.
pub fn embedding_scatter_source(
  plan : @effects.EmbeddingScatterProgram,
  symbol~ : String,
) -> EmbeddingScatterSource raise {
  guard !symbol.is_empty() &&
    symbol.length() <= 128 &&
    symbol[0] is ('a'..<='z' | 'A'..<='Z' | '_') &&
    symbol.all(c => c is ('a'..<='z' | 'A'..<='Z' | '0'..<='9' | '_')) else {
    fail("invalid embedding scatter symbol")
  }
  let (width, rows, positions, features) = plan.geometry()
  let text =
    $|#include <stdint.h>
    $|extern "C" __global__ void \{symbol}(const int32_t* counts,const int32_t* positions,const int32_t* feature_map,const int32_t* feature_count,const uint16_t* features,uint16_t* residual){
    $| const int row=blockIdx.x,live=counts[3],n=feature_count[0];
    $| if(counts[0]!=1||live<=0||live>\{rows}||row>=live||n<=0||n>\{features})return;
    $| const int position=positions[row];if(position<0||position>=\{positions})return;
    $| const int feature=feature_map[position];if(feature<0||feature>=n)return;
    $| for(unsigned long long column=threadIdx.x;column<\{width}ULL;column+=blockDim.x)
    $|  residual[(unsigned long long)row*\{width}ULL+column]=features[(unsigned long long)feature*\{width}ULL+column];
    $|}
  { plan, symbol, bytes: @utf8.encode(text), }
}

///|
pub fn EmbeddingScatterSource::source_bytes(self : Self) -> Bytes {
  self.bytes
}

///|
pub fn EmbeddingScatterSource::symbol(self : Self) -> String {
  self.symbol
}

///|
pub fn EmbeddingScatterSource::plan(
  self : Self,
) -> @effects.EmbeddingScatterProgram {
  self.plan
}

///|
pub fn EmbeddingScatterSource::launch(self : Self) -> (Int32, Int32, Int32) {
  let (_, rows, _, _) = self.plan.geometry()
  let (_, owners, _) = self.plan.ownership().geometry()
  (rows, 1, owners)
}

///|
pub fn EmbeddingScatterSource::parameter_count(_self : Self) -> Int32 {
  6
}
