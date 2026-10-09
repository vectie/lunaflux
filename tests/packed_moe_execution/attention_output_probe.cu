#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "output_kernels.cuh"

static void check(cudaError_t result) {
  if (result != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(result)); std::exit(2); }
}
static void require(bool yes, const char *message) {
  if (!yes) { std::fprintf(stderr, "%s\n", message); std::exit(3); }
}
static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float fp8(unsigned char x) {
  int exponent = (x >> 3) & 15, mantissa = x & 7;
  float value = exponent == 0 ? std::ldexp((float)mantissa, -9) : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
  return (x & 128) ? -value : value;
}
static float round_fp8(float x) {
  bool negative = x < 0; float absolute = std::fmin(448.0f, std::fabs(x));
  int best = 0; float distance = absolute;
  for (int bits = 1; bits <= 126; ++bits) {
    float delta = std::fabs(fp8((unsigned char)bits) - absolute);
    if (delta < distance || (delta == distance && (bits & 1) == 0)) { best = bits; distance = delta; }
  }
  return negative ? -fp8((unsigned char)best) : fp8((unsigned char)best);
}
static float frequency(int pair, bool yarn) {
  float base = 1.0f / std::pow(yarn ? 160000.0f : 10000.0f, (float)(pair * 2) / 64.0f);
  if (!yarn) return base;
  double fast = 64.0 * std::log(65536.0 / (32.0 * 6.2831853071795864769)) / (2.0 * std::log(160000.0));
  double slow = 64.0 * std::log(65536.0 / 6.2831853071795864769) / (2.0 * std::log(160000.0));
  float low = (float)std::fmax(0.0, std::floor(fast)), high = (float)std::fmin(63.0, std::ceil(slow));
  float span = high + (-low + (high == low ? 0.001f : 0.0f));
  float ramp = std::fmin(1.0f, std::fmax(0.0f, ((float)pair - low) / span)), smooth = 1.0f - ramp;
  return (base / 16.0f) * (1.0f - smooth) + base * smooth;
}
static float compare(const std::vector<__nv_bfloat16>& actual, const std::vector<float>& expected, int live, float tolerance) {
  float maximum = 0;
  for (int i = 0; i < live; ++i) {
    float value = __bfloat162float(actual[i]); float delta = std::fabs(value - expected[i]);
    require(std::isfinite(value) && delta <= tolerance + std::fabs(expected[i]) * 0.01f, "numerical mismatch");
    maximum = std::fmax(maximum, delta);
  }
  return maximum;
}
int main() {
  constexpr int rows = 3, heads = 2, width = 128, groups = 2, rank = 128, hidden = 128;
  int *counts, *positions; __nv_bfloat16 *input, *inverse, *middle, *result;
  unsigned char *wa, *sa, *wb, *sb;
  check(cudaMalloc(&counts, 20)); check(cudaMalloc(&positions, rows * 4));
  check(cudaMalloc(&input, rows * heads * width * 2)); check(cudaMalloc(&inverse, rows * heads * width * 2));
  check(cudaMalloc(&middle, rows * groups * rank * 2)); check(cudaMalloc(&result, rows * hidden * 2));
  check(cudaMalloc(&wa, groups * rank * width)); check(cudaMalloc(&sa, 2));
  check(cudaMalloc(&wb, hidden * groups * rank)); check(cudaMalloc(&sb, 2));
  std::vector<unsigned char> aw(groups * rank * width), bw(hidden * groups * rank);
  unsigned char as[2] = {126, 127}, bs[2] = {126, 125};
  const unsigned char values[] = {0x10, 0x98, 0x20, 0xa8, 0x30, 0xb8, 0x00, 0x08};
  for (int i = 0; i < (int)aw.size(); ++i) aw[i] = values[(i * 13 + i / 128) % 8];
  for (int i = 0; i < (int)bw.size(); ++i) bw[i] = values[(i * 3 + i / 256) % 8];
  check(cudaMemcpy(wa, aw.data(), aw.size(), cudaMemcpyHostToDevice)); check(cudaMemcpy(sa, as, 2, cudaMemcpyHostToDevice));
  check(cudaMemcpy(wb, bw.data(), bw.size(), cudaMemcpyHostToDevice)); check(cudaMemcpy(sb, bs, 2, cudaMemcpyHostToDevice));
  for (bool yarn : {false, true}) for (int live : {1, 2, 3, 0}) {
    int control[5] = {0, 0, 1, live, live}; int pos[rows] = {0, 65537, 1048575};
    std::vector<__nv_bfloat16> x(rows * heads * width), ri(x.size(), __float2bfloat16_rn(19)), rm(rows * groups * rank, __float2bfloat16_rn(19)), ro(rows * hidden, __float2bfloat16_rn(19));
    for (int i = 0; i < (int)x.size(); ++i) x[i] = __float2bfloat16_rn((float)((i * 17) % 61 - 30) / 64.0f);
    check(cudaMemcpy(counts, control, 20, cudaMemcpyHostToDevice)); check(cudaMemcpy(positions, pos, sizeof(pos), cudaMemcpyHostToDevice));
    check(cudaMemcpy(input, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(inverse, ri.data(), ri.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(middle, rm.data(), rm.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(result, ro.data(), ro.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> ei(x.size()), em(rm.size()), eo(ro.size());
    for (int row = 0; row < live; ++row) for (int head = 0; head < heads; ++head) {
      int base = (row * heads + head) * width;
      for (int col = 0; col < 64; ++col) ei[base + col] = __bfloat162float(x[base + col]);
      for (int pair = 0; pair < 32; ++pair) {
        float angle = (float)pos[row] * frequency(pair, yarn), c = std::cos(angle), s = -std::sin(angle);
        int offset = base + 64 + pair * 2; float left = __bfloat162float(x[offset]), right = __bfloat162float(x[offset + 1]);
        ei[offset] = bf(left * c - right * s); ei[offset + 1] = bf(left * s + right * c);
      }
    }
    for (int row = 0; row < live; ++row) for (int group = 0; group < groups; ++group) for (int col = 0; col < rank; ++col) {
      float sum = 0; int weight_row = group * rank + col;
      for (int inner = 0; inner < width; ++inner) {
        float parameter = bf(fp8(aw[weight_row * width + inner]) * std::ldexp(1.0f, as[weight_row / 128] - 127));
        sum = sum + ei[(row * groups + group) * width + inner] * parameter;
      }
      em[(row * groups + group) * rank + col] = bf(sum);
    }
    for (int row = 0; row < live; ++row) for (int col = 0; col < hidden; ++col) {
      float sum = 0;
      for (int block = 0; block < 2; ++block) {
        float maximum = 0.0001f;
        for (int j = 0; j < 128; ++j) maximum = std::fmax(maximum, std::fabs(em[row * 256 + block * 128 + j]));
        float requested = maximum * 0.002232142857142857f, scale = std::exp2(std::ceil(std::log2(requested))), partial = 0;
        for (int j = 0; j < 128; ++j) {
          int inner = block * 128 + j;
          partial = partial + round_fp8(em[row * 256 + inner] / scale) * fp8(bw[col * 256 + inner]);
        }
        sum = sum + partial * (scale * std::ldexp(1.0f, bs[block] - 127));
      }
      eo[row * hidden + col] = bf(sum);
    }
    auto run = [&]() {
      if (yarn) {
        output_yarn_inverse<<<dim3(rows, heads), 256>>>(counts, positions, (const uint16_t *)input, (uint16_t *)inverse);
        output_yarn_grouped<<<dim3(rows, groups), 256>>>(counts, inverse, wa, sa, middle);
        output_yarn_output<<<dim3(rows, 1), 256>>>(counts, middle, wb, sb, result);
      } else {
        output_inverse<<<dim3(rows, heads), 256>>>(counts, positions, (const uint16_t *)input, (uint16_t *)inverse);
        output_grouped<<<dim3(rows, groups), 256>>>(counts, inverse, wa, sa, middle);
        output_output<<<dim3(rows, 1), 256>>>(counts, middle, wb, sb, result);
      }
      check(cudaGetLastError()); check(cudaDeviceSynchronize());
    };
    run();
    check(cudaMemcpy(ri.data(), inverse, ri.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(rm.data(), middle, rm.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(ro.data(), result, ro.size() * 2, cudaMemcpyDeviceToHost));
    float di = compare(ri, ei, live * heads * width, 0.0078125f), dm = compare(rm, em, live * groups * rank, 0.03125f), dout = compare(ro, eo, live * hidden, 0.0625f);
    for (int i = live * heads * width; i < (int)ri.size(); ++i) require(__bfloat162float(ri[i]) == 19, "inactive inverse overwritten");
    for (int i = live * groups * rank; i < (int)rm.size(); ++i) require(__bfloat162float(rm[i]) == 19, "inactive grouped overwritten");
    for (int i = live * hidden; i < (int)ro.size(); ++i) require(__bfloat162float(ro[i]) == 19, "inactive output overwritten");
    for (int repeat = 0; repeat < 4; ++repeat) {
      run(); std::vector<__nv_bfloat16> replay(ro.size()); check(cudaMemcpy(replay.data(), result, replay.size() * 2, cudaMemcpyDeviceToHost));
      require(std::memcmp(replay.data(), ro.data(), ro.size() * 2) == 0, "nondeterministic grouped output");
    }
    std::printf("yarn=%d live=%d inverse_max=%.9g grouped_max=%.9g output_max=%.9g\n", (int)yarn, live, di, dm, dout);
  }
  check(cudaFree(sb)); check(cudaFree(wb)); check(cudaFree(sa)); check(cudaFree(wa));
  check(cudaFree(result)); check(cudaFree(middle)); check(cudaFree(inverse)); check(cudaFree(input));
  check(cudaFree(positions)); check(cudaFree(counts)); std::puts("grouped output correctness and deterministic replay passed");
}
