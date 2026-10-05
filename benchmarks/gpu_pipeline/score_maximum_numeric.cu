// Offline CUDA numeric-law probe; never linked into serving. Tests the exact
// fmaxf primitive used by terminal lowering, including nonfinite/zero edges.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr int cases = 4096, width = 16;

__global__ void compare_maxima(const unsigned *input, unsigned *result) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= cases) return;
  float ordered = __uint_as_float(0xff800000U), tree[width];
  #pragma unroll
  for (int i = 0; i < width; ++i) {
    tree[i] = __uint_as_float(input[row * width + i]);
    ordered = fmaxf(ordered, tree[i]);
  }
  #pragma unroll
  for (int n = width; n > 1; n /= 2) {
    #pragma unroll
    for (int i = 0; i < n / 2; ++i) tree[i] = fmaxf(tree[2 * i], tree[2 * i + 1]);
  }
  result[row * 2] = __float_as_uint(ordered);
  result[row * 2 + 1] = __float_as_uint(fmaxf(__uint_as_float(0xff800000U), tree[0]));
}

static bool checked(cudaError_t status) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
  return false;
}

int main() {
  const unsigned edges[] = {0U, 0x80000000U, 0x7f800000U, 0xff800000U,
    0x7fc00001U, 0xffc00123U, 1U, 0x80000001U, 0x3f800000U, 0xbf800000U};
  std::vector<unsigned> input(cases * width), output(cases * 2);
  unsigned seed = 0x12345678U;
  for (int row = 0; row < cases; ++row) {
    for (int i = 0; i < width; ++i) {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      input[row * width + i] = row < 10 ? edges[row] :
        (row < 2048 ? edges[seed % 10] : seed);
    }
  }
  unsigned *device_input = nullptr, *device_output = nullptr;
  bool ok = checked(cudaMalloc(&device_input, input.size() * sizeof(unsigned)));
  if (ok) ok = checked(cudaMalloc(&device_output, output.size() * sizeof(unsigned)));
  if (ok) ok = checked(cudaMemcpy(device_input, input.data(), input.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  if (ok) {
    compare_maxima<<<cases / 128, 128>>>(device_input, device_output);
    ok = checked(cudaGetLastError()) && checked(cudaDeviceSynchronize());
  }
  if (ok) ok = checked(cudaMemcpy(output.data(), device_output, output.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  if (device_output) ok = checked(cudaFree(device_output)) && ok;
  if (device_input) ok = checked(cudaFree(device_input)) && ok;
  if (!ok) return 2;
  for (int row = 0; row < cases; ++row) {
    if (output[row * 2] != output[row * 2 + 1]) {
      std::fprintf(stderr, "row=%d ordered=%08x tree=%08x\n", row, output[row * 2], output[row * 2 + 1]);
      return 1;
    }
  }
  std::puts("correctness=passed bitwise_maximum=true cases=4096 nonfinite=true signed_zero=true");
  return 0;
}
