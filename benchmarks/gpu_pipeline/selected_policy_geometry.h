#pragma once
#include <algorithm>
#include <stdexcept>

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
