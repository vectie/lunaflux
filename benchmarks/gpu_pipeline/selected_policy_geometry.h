#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <map>

// The compiler exports standalone decode chains separately from the combined
// serving module. Adapt names only: preserve both entries, the physical row
// envelope and the numerical law. Never reinterpret prefill metadata as CSR.
inline std::map<std::string,std::string> selected_decode_chain_recipe(
    const std::map<std::string,std::string>& recipe, bool decode, bool partitioned) {
  auto out=recipe;
  auto family=out.find("family");
  if(family==out.end() || family->second!="paged-attention-decode-functional-partitioned-tile")
    return out;
  if(!decode || !partitioned ||
     out.at("schema")!="lunaflux-attention-tile-compiler-partitioned-cuda-aot-candidate.v1" ||
     out.at("measurement_boundary")!="standalone-partial-merge-v1" ||
     out.at("merge_shared_memory_bytes")!="0")
    throw std::invalid_argument("decode chain contract");
  for(const char* key:{"merge_function_symbol","merge_grid","block","workspace_bytes","numeric_law"})
    if(out.at(key).empty())throw std::invalid_argument("incomplete decode chain");
  out["function_symbol"]=out.at("partial_function_symbol");
  out["grid"]=out.at("partial_grid");
  out["shared_memory_bytes"]=out.at("partial_shared_memory_bytes");
  return out;
}

// Offline mirror of device_step Query{Token,Tile,Metadata}CappedGridX.
// Metadata capacity uses profile rows, not the observed equal-length row
// distribution: capture must also cover independent row tails in that bucket.
inline int selected_bucket_tokens(int tokens, int maximum) {
  if (tokens <= 0 || maximum < tokens) throw std::invalid_argument("bucket");
  int bound = 1;
  while (bound < tokens && bound < maximum) bound = std::min(maximum, bound * 2);
  return bound;
}
// Captured graphs may deliberately use a minimum bucket larger than the
// logical batch. An explicit traced bound reproduces that launch envelope.
inline int selected_capture_bound(int tokens, int maximum, int traced = 0) {
  if (!traced) return selected_bucket_tokens(tokens, maximum);
  if (tokens <= 0 || traced < tokens || traced > maximum || (traced & (traced - 1)))
    throw std::invalid_argument("traced capture bound");
  return traced;
}
// Mirror device_step's decode capture domain, not the prefill power-of-two
// domain. C2/C4 replay an eight-row graph, including its inactive rows.
inline int selected_decode_capture_bound(int rows, int maximum, int traced = 0) {
  if (rows <= 0 || rows > maximum) throw std::invalid_argument("decode bucket");
  for (int candidate : {1, 8, 16, 32}) {
    const int bound = std::min(candidate, maximum);
    if (traced ? bound == traced : bound >= rows) {
      if (bound < rows) throw std::invalid_argument("traced decode bound");
      return bound;
    }
  }
  if (traced && traced != maximum) throw std::invalid_argument("traced decode bound");
  return maximum;
}
inline int selected_grid_x(int envelope, int bucket_tokens, int profile_rows,
                           int query_tile, bool metadata, bool decode) {
  if (envelope <= 0 || bucket_tokens <= 0 || profile_rows <= 0 || query_tile <= 0)
    throw std::invalid_argument("geometry");
  const int rows = std::min(profile_rows, bucket_tokens);
  const int tiles = decode ? bucket_tokens : metadata
      ? rows + (bucket_tokens - rows) / query_tile
      : 1 + (bucket_tokens - 1) / query_tile;
  return std::min(envelope, tiles);
}
// Partial and merge are distinct entries. Legacy recipes deliberately use
// one common block; explicit merge geometry must not inherit a changed partial.
inline int selected_merge_block(int partial, int declared = 0) {
  if (partial <= 0 || partial > 1024 || declared < 0 || declared > 1024)
    throw std::invalid_argument("merge block");
  return declared ? declared : partial;
}

// Mixed work is not necessarily balanced: the scheduler may fill the token
// budget from one prefill row while the other rows decode. Keep that physical
// domain explicit in offline calibration rather than silently using R-1.
inline int selected_mixed_prefill_rows(int rows, int declared = 0) {
  const int count = declared ? declared : rows - 1;
  if (rows < 2 || count < 1 || count >= rows)
    throw std::invalid_argument("mixed prefill rows");
  return count;
}

// Diagnostic logical domain: query length and prior history for every row.
// Validate before allocating or touching CUDA. Mixed rows are prefill first,
// decode last, matching the runtime ABI rather than a balanced proxy.
inline std::vector<std::pair<int,int>> selected_row_work(
    const std::string& text, int rows, int tokens, int prefill_rows,
    int page_tokens, int page_capacity) {
  std::vector<std::pair<int,int>> result;
  std::istringstream input(text); std::string row;
  long long sum=0, pages=0;
  if(rows<2 || tokens<rows || prefill_rows<1 || prefill_rows>=rows || page_tokens<1 || page_capacity<1)
    throw std::invalid_argument("row work domain");
  while(std::getline(input,row,',')) {
    std::istringstream values(row); long long q,p; char colon,extra;
    if(!(values>>q>>colon>>p) || colon!=':' || (values>>extra) || q<1 || p<0 ||
       q>tokens || p>2147483647LL-q || int(result.size())>=rows ||
       (int(result.size())>=prefill_rows && q!=1))
      throw std::invalid_argument("row work entry");
    sum+=q; pages+=(q+p+page_tokens-1)/page_tokens;
    result.emplace_back(int(q),int(p));
  }
  if(text.empty() || text.back()==',' || int(result.size())!=rows || sum!=tokens || pages>page_capacity)
    throw std::invalid_argument("row work extent");
  return result;
}
