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
