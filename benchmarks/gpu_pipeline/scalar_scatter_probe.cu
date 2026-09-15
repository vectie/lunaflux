// Standalone diagnostic: compile beside the exact exported scalar-scatter.cu.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "scalar-scatter.cu"

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(2);
  }
}

int main() {
  int *counts, *offsets;
  __nv_bfloat16 *input, *weights, *output;
  check(cudaMalloc(&counts, 5 * sizeof(int)));
  check(cudaMalloc(&offsets, 3 * sizeof(int)));
  check(cudaMalloc(&input, 16 * sizeof(__nv_bfloat16)));
  check(cudaMalloc(&weights, 64 * sizeof(__nv_bfloat16)));
  check(cudaMalloc(&output, 64 * sizeof(__nv_bfloat16)));
  __nv_bfloat16 x[16], w[64], y[64];
  for (int row = 0; row < 4; ++row)
    for (int k = 0; k < 4; ++k) x[row * 4 + k] = __float2bfloat16(float(row + k + 1));
  for (int c = 0; c < 16; ++c)
    for (int k = 0; k < 4; ++k) w[c * 4 + k] = __float2bfloat16(float((c % 3) + 1));
  check(cudaMemcpy(input, x, sizeof(x), cudaMemcpyHostToDevice));
  check(cudaMemcpy(weights, w, sizeof(w), cudaMemcpyHostToDevice));
  // Noncontiguous ends, contiguous decode, one long row, and invalid row.
  const int rows[][3] = {{0, 2, 4}, {0, 1, 4}, {0, 1, 2}, {0, 4, 4}, {0, 0, 4}};
  for (int trial = 0; trial < 5; ++trial) {
    int nrows = trial == 3 ? 1 : 2;
    int n = trial == 2 ? 2 : 4;
    int hcounts[5] = {nrows, 0, nrows, n, 1};
    for (auto &value : y) value = __float2bfloat16(-99.0f);
    check(cudaMemcpy(counts, hcounts, sizeof(hcounts), cudaMemcpyHostToDevice));
    check(cudaMemcpy(offsets, rows[trial], sizeof(rows[trial]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(output, y, sizeof(y), cudaMemcpyHostToDevice));
    selected_head_probe<<<1, 256>>>(counts, offsets, input, weights, output);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(y, output, sizeof(y), cudaMemcpyDeviceToHost));
    for (int row = 0; row < 4; ++row) {
      bool selected = false;
      for (int r = 0; r < nrows; ++r) selected |= row == rows[trial][r + 1] - 1;
      for (int c = 0; c < 16; ++c) {
        float expected = selected ? float((4 * row + 10) * ((c % 3) + 1)) : -99.0f;
        if (__bfloat162float(y[row * 16 + c]) != expected) {
          std::fprintf(stderr, "scatter mismatch trial=%d row=%d column=%d\n", trial, row, c);
          return 3;
        }
      }
    }
  }
  check(cudaFree(output)); check(cudaFree(weights)); check(cudaFree(input));
  check(cudaFree(offsets)); check(cudaFree(counts));
  std::puts("passed: 5 selected-row cases, selected values and untouched rows");
}
