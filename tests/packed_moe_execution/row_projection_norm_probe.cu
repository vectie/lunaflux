// Independent CPU oracle for packed projection -> BF16 -> learned RMSNorm.
// A component numerical fixture, not whole-model correctness or performance.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "main.cuh"

static void check(cudaError_t result) {
  if (result != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(result)); std::exit(2); }
}
template<class T> static T *allocate(size_t count) {
  T *result = nullptr;
  check(cudaMallocManaged(&result, count * sizeof(T)));
  return result;
}
static float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
static float e4(unsigned char bits) { __nv_fp8_e4m3 v; v.__x = bits; return (float)v; }
static float quantize(float v) {
  v = std::max(-448.f, std::min(448.f, v));
  if (v == 0.f) v = 0.f;
  return e4(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

int main() {
  constexpr int rows = 3, in = 256, out = 128;
  auto counts = allocate<int>(5);
  auto input = allocate<__nv_bfloat16>(rows * in);
  auto weights = allocate<unsigned char>(out * in);
  auto scales = allocate<unsigned char>(2);
  auto norm = allocate<__nv_bfloat16>(out);
  auto projected = allocate<__nv_bfloat16>(rows * out);
  auto normalized = allocate<__nv_bfloat16>(rows * out);
  std::vector<float> expected_projection(rows * out), expected_norm(rows * out);
  std::vector<__nv_bfloat16> first_projection(rows * out), first_norm(rows * out);
  for (int i = 0; i < out * in; ++i)
    weights[i] = (unsigned char)(16 + (i * 7) % 32 + ((i % 3) == 0 ? 128 : 0));
  for (int i = 0; i < out; ++i) norm[i] = __float2bfloat16_rn(0.75f + float(i % 9) * 0.03125f);
  int cases = 0;
  float worst_projection = 0.f, worst_norm = 0.f;
  for (int mode : {0, 1, 2}) for (int live : {-1, 0, 1, 2, 3, 4}) {
    std::memset(counts, 0, 5 * sizeof(int));
    counts[3] = live;
    scales[0] = mode == 2 ? 119 : 124;
    scales[1] = mode == 2 ? 134 : 125;
    for (int i = 0; i < rows * in; ++i)
      input[i] = __float2bfloat16_rn(mode == 0 ? 0.f : float((i * 13) % 31 - 15) * (mode == 2 ? 0.001953125f : 0.125f));
    std::fill(expected_projection.begin(), expected_projection.end(), 19.f);
    std::fill(expected_norm.begin(), expected_norm.end(), 19.f);
    if (live > 0 && live <= rows) {
      for (int row = 0; row < live; ++row) {
        for (int col = 0; col < out; ++col) {
          float sum = 0.f;
          for (int block = 0; block < 2; ++block) {
            float maximum = 0.0001f;
            for (int k = 0; k < 128; ++k)
              maximum = std::max(maximum, std::fabs(__bfloat162float(input[row * in + block * 128 + k])));
            int exponent = 0;
            float fraction = std::frexp(maximum / 448.f, &exponent);
            float activation_scale = std::ldexp(1.f, fraction == 0.5f ? exponent - 1 : exponent);
            float partial = 0.f;
            for (int k = 0; k < 128; ++k) {
              float activation = quantize(__bfloat162float(input[row * in + block * 128 + k]) / activation_scale);
              partial = partial + activation * e4(weights[col * in + block * 128 + k]);
            }
            float weight_scale = std::ldexp(1.f, int(scales[block]) - 127);
            sum = sum + partial * (activation_scale * weight_scale);
          }
          expected_projection[row * out + col] = bf(sum);
        }
        float square_sum = 0.f;
        for (int col = 0; col < out; ++col) {
          float value = expected_projection[row * out + col];
          square_sum = square_sum + value * value;
        }
        float inverse = 1.f / std::sqrt(square_sum / out + 1.e-6f);
        for (int col = 0; col < out; ++col)
          expected_norm[row * out + col] = bf((expected_projection[row * out + col] * inverse) * __bfloat162float(norm[col]));
      }
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
      for (int i = 0; i < rows * out; ++i) projected[i] = normalized[i] = __float2bfloat16_rn(19.f);
      main_fixture_projection<<<dim3(rows, 1), 256>>>(counts, input, weights, scales, projected);
      main_fixture_norm<<<rows, 256>>>(counts, projected, norm, normalized);
      check(cudaGetLastError());
      check(cudaDeviceSynchronize());
      for (int i = 0; i < rows * out; ++i) {
        float p = __bfloat162float(projected[i]), n = __bfloat162float(normalized[i]);
        float p_error = std::fabs(p - expected_projection[i]), n_error = std::fabs(n - expected_norm[i]);
        worst_projection = std::max(worst_projection, p_error);
        worst_norm = std::max(worst_norm, n_error);
        const bool active = live > 0 && live <= rows && i < live * out;
        if (!std::isfinite(p) || !std::isfinite(n) ||
            p_error != 0.f || (active ? n_error > 0.015625f + 0.005f * std::fabs(expected_norm[i]) : n_error != 0.f)) {
          std::fprintf(stderr, "mode=%d live=%d index=%d projected=%g expected=%g normalized=%g expected=%g\n",
            mode, live, i, p, expected_projection[i], n, expected_norm[i]);
          return 3;
        }
      }
      if (repeat == 0) {
        std::memcpy(first_projection.data(), projected, rows * out * sizeof(__nv_bfloat16));
        std::memcpy(first_norm.data(), normalized, rows * out * sizeof(__nv_bfloat16));
      } else if (std::memcmp(first_projection.data(), projected, rows * out * sizeof(__nv_bfloat16)) ||
                 std::memcmp(first_norm.data(), normalized, rows * out * sizeof(__nv_bfloat16))) return 4;
      ++cases;
    }
  }
  check(cudaFree(normalized)); check(cudaFree(projected)); check(cudaFree(norm));
  check(cudaFree(scales)); check(cudaFree(weights)); check(cudaFree(input)); check(cudaFree(counts));
  check(cudaDeviceReset());
  std::printf("row projection norm: %d GPU cases passed; maxabs_projection=%g maxabs_norm=%g; deterministic, inactive/invalid rows untouched, all allocations released\n",
    cases, worst_projection, worst_norm);
}
