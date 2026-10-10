#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "capture.cuh"

static void checked(cudaError_t code) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(code));
    std::exit(2);
  }
}

int main() {
  constexpr int rows = 3, streams = 4, hidden = 257, segments = 3;
  const int input_size = rows * streams * hidden;
  const int output_size = rows * segments * hidden;
  std::vector<__nv_bfloat16> input(input_size), output(output_size), expected(output_size);
  int32_t *counts = nullptr;
  __nv_bfloat16 *source = nullptr, *destination = nullptr;
  checked(cudaMalloc(&counts, 5 * sizeof(int32_t)));
  checked(cudaMalloc(&source, input_size * sizeof(__nv_bfloat16)));
  checked(cudaMalloc(&destination, output_size * sizeof(__nv_bfloat16)));
  const __nv_bfloat16 sentinel = __float2bfloat16(31.5f);
  int cases = 0;
  for (int repeat = 0; repeat < 32; ++repeat) {
    for (int live : {-1, 0, 1, 2, 3, 4}) {
      int32_t host_counts[5] = {0, 0, 0, live, 0};
      checked(cudaMemcpy(counts, host_counts, sizeof(host_counts), cudaMemcpyHostToDevice));
      std::fill(expected.begin(), expected.end(), sentinel);
      checked(cudaMemcpy(destination, expected.data(), output_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
      for (int segment = 0; segment < segments; ++segment) {
        for (int row = 0; row < rows; ++row) {
          for (int stream = 0; stream < streams; ++stream) {
            for (int column = 0; column < hidden; ++column) {
              const float value = float((row * 11 + stream * 7 + column * 3 + segment * 5 + repeat) % 19 - 9) * 0.1328125f;
              input[(row * streams + stream) * hidden + column] = __float2bfloat16(value);
            }
          }
        }
        if (live > 0 && live <= rows) {
          for (int row = 0; row < live; ++row) {
            for (int column = 0; column < hidden; ++column) {
              float sum = 0.0f;
              for (int stream = 0; stream < streams; ++stream)
                sum += __bfloat162float(input[(row * streams + stream) * hidden + column]);
              expected[(row * segments + segment) * hidden + column] = __float2bfloat16(sum / float(streams));
            }
          }
        }
        checked(cudaMemcpy(source, input.data(), input_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        if (segment == 0) capture_segment_0<<<4, 256>>>(counts, source, destination);
        if (segment == 1) capture_segment_1<<<4, 256>>>(counts, source, destination);
        if (segment == 2) capture_segment_2<<<4, 256>>>(counts, source, destination);
        checked(cudaGetLastError());
      }
      checked(cudaMemcpy(output.data(), destination, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      for (int i = 0; i < output_size; ++i) {
        if (__bfloat162float(output[i]) != __bfloat162float(expected[i])) {
          std::fprintf(stderr, "capture mismatch: repeat=%d live=%d element=%d\n", repeat, live, i);
          return 3;
        }
      }
      ++cases;
    }
  }
  checked(cudaFree(destination));
  checked(cudaFree(source));
  checked(cudaFree(counts));
  checked(cudaDeviceReset());
  std::printf("stream mean capture: %d GPU cases passed; all segments, BF16 rounding, inactive/invalid rows and replay\n", cases);
  return 0;
}
