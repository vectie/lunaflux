#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <vector>

namespace {
constexpr int kQueryHeads = 2;
constexpr int kHeadDimension = 2;
constexpr int kTokensPerPage = 2;
constexpr int kQkvWidth = 8;
constexpr int kOutputWidth = 4;
constexpr int kPageStrideValues = 128;
constexpr int kPhysicalPages = 4;

uint16_t to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + bias) >> 16U);
}

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool cuda_ok(CUresult result, const char *stage) {
  if (result == CUDA_SUCCESS) return true;
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s: %s: %s\n", stage, name ? name : "CUDA_ERROR",
               message ? message : "unknown");
  return false;
}

size_t cache_index(int physical_page, int position, int component) {
  return static_cast<size_t>(physical_page) * kPageStrideValues +
         static_cast<size_t>(position % kTokensPerPage) * kHeadDimension + component;
}

float live_or_cache(const std::vector<uint16_t> &qkv,
                    const std::vector<uint16_t> &cache,
                    const int *positions, const int *row_offsets,
                    const int *page_offsets, const int *page_indices, int row,
                    int key_position, int component, int qkv_component_offset) {
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  const int chunk_start = positions[row_start];
  if (key_position >= chunk_start) {
    const int local = row_start + key_position - chunk_start;
    if (local < row_end && positions[local] == key_position) {
      return from_bf16(qkv[local * kQkvWidth + qkv_component_offset + component]);
    }
  }
  const int page_slot = page_offsets[row] + key_position / kTokensPerPage;
  return from_bf16(cache[cache_index(page_indices[page_slot], key_position, component)]);
}

std::vector<float> reference(const std::vector<uint16_t> &qkv,
                             const std::vector<uint16_t> &key_cache,
                             const std::vector<uint16_t> &value_cache,
                             const int *positions, const int *row_offsets,
                             const int *page_offsets, const int *page_indices) {
  std::vector<float> output(3 * kOutputWidth, 0.0f);
  const float scale = 1.0f / std::sqrt(2.0f);
  for (int token = 0; token < 3; ++token) {
    const int row = token < row_offsets[1] ? 0 : 1;
    const int query_position = positions[token];
    for (int head = 0; head < kQueryHeads; ++head) {
      float scores[3] = {};
      float maximum = -std::numeric_limits<float>::infinity();
      for (int key_position = 0; key_position <= query_position; ++key_position) {
        float dot = 0.0f;
        for (int d = 0; d < kHeadDimension; ++d) {
          const float query = from_bf16(qkv[token * kQkvWidth + head * 2 + d]);
          const float key = live_or_cache(qkv, key_cache, positions, row_offsets,
                                          page_offsets, page_indices, row,
                                          key_position, d, 4);
          dot += query * key;
        }
        scores[key_position] = dot * scale;
        maximum = std::fmax(maximum, scores[key_position]);
      }
      float denominator = 0.0f;
      for (int key_position = 0; key_position <= query_position; ++key_position)
        denominator += std::exp(scores[key_position] - maximum);
      for (int d = 0; d < kHeadDimension; ++d) {
        float sum = 0.0f;
        for (int key_position = 0; key_position <= query_position; ++key_position) {
          const float value = live_or_cache(qkv, value_cache, positions, row_offsets,
                                            page_offsets, page_indices, row,
                                            key_position, d, 6);
          sum += std::exp(scores[key_position] - maximum) * value / denominator;
        }
        output[token * kOutputWidth + head * 2 + d] = sum;
      }
    }
  }
  return output;
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: physical_probe CUBIN SYMBOL\n");
    return 2;
  }
  const int counts[5] = {1, 1, 2, 3, 3};
  const int positions[3] = {0, 1, 2};
  const int row_offsets[3] = {0, 2, 3};
  const int sequence_lengths[2] = {2, 3};
  const int page_offsets[3] = {0, 1, 3};
  const int page_indices[3] = {0, 1, 2};
  const float qkv_float[24] = {
      1, 0, 0, 1, 1, 0, 1, 2,
      0, 1, 1, 0, 0, 1, 2, 1,
      1, 1, -1, 1, 1, 1, 7, 8,
  };
  std::vector<uint16_t> qkv(24);
  for (int i = 0; i < 24; ++i) qkv[i] = to_bf16(qkv_float[i]);
  std::vector<uint16_t> key_cache(kPhysicalPages * kPageStrideValues, 0);
  std::vector<uint16_t> value_cache(kPhysicalPages * kPageStrideValues, 0);
  key_cache[cache_index(1, 0, 0)] = to_bf16(1);
  key_cache[cache_index(1, 0, 1)] = to_bf16(0);
  value_cache[cache_index(1, 0, 0)] = to_bf16(3);
  value_cache[cache_index(1, 0, 1)] = to_bf16(4);
  key_cache[cache_index(1, 1, 0)] = to_bf16(0);
  key_cache[cache_index(1, 1, 1)] = to_bf16(1);
  value_cache[cache_index(1, 1, 0)] = to_bf16(5);
  value_cache[cache_index(1, 1, 1)] = to_bf16(6);
  const auto expected = reference(qkv, key_cache, value_cache, positions, row_offsets,
                                  page_offsets, page_indices);

  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUdeviceptr allocations[10] = {};
  const size_t sizes[10] = {sizeof(counts), sizeof(positions), sizeof(row_offsets),
      sizeof(sequence_lengths), sizeof(page_offsets), sizeof(page_indices),
      qkv.size() * sizeof(uint16_t), expected.size() * sizeof(uint16_t),
      key_cache.size() * sizeof(uint16_t), value_cache.size() * sizeof(uint16_t)};
  if (!cuda_ok(cuInit(0), "cuInit") ||
      !cuda_ok(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !cuda_ok(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate") ||
      !cuda_ok(cuModuleLoad(&module, argv[1]), "cuModuleLoad") ||
      !cuda_ok(cuModuleGetFunction(&function, module, argv[2]), "cuModuleGetFunction"))
    return 1;
  for (int i = 0; i < 10; ++i)
    if (!cuda_ok(cuMemAlloc(&allocations[i], sizes[i]), "cuMemAlloc")) return 1;
  const void *inputs[10] = {counts, positions, row_offsets, sequence_lengths,
      page_offsets, page_indices, qkv.data(), nullptr, key_cache.data(), value_cache.data()};
  for (int i : {0, 1, 2, 3, 4, 5, 6, 8, 9})
    if (!cuda_ok(cuMemcpyHtoD(allocations[i], inputs[i], sizes[i]), "cuMemcpyHtoD")) return 1;
  void *arguments[10];
  for (int i = 0; i < 10; ++i) arguments[i] = &allocations[i];
  if (!cuda_ok(cuLaunchKernel(function, 3, 2, 1, 1, 1, 1, 0, nullptr, arguments, nullptr),
               "cuLaunchKernel") ||
      !cuda_ok(cuCtxSynchronize(), "cuCtxSynchronize")) return 1;
  std::vector<uint16_t> actual(expected.size());
  if (!cuda_ok(cuMemcpyDtoH(actual.data(), allocations[7], sizes[7]), "copy output") ||
      !cuda_ok(cuMemcpyDtoH(key_cache.data(), allocations[8], sizes[8]), "copy key cache") ||
      !cuda_ok(cuMemcpyDtoH(value_cache.data(), allocations[9], sizes[9]), "copy value cache"))
    return 1;
  float maximum_error = 0.0f;
  for (size_t i = 0; i < expected.size(); ++i) {
    const float error = std::fabs(from_bf16(actual[i]) - expected[i]);
    maximum_error = std::fmax(maximum_error, error);
    if (!std::isfinite(error) || error > 0.05f) {
      std::fprintf(stderr, "output mismatch index=%zu expected=%g actual=%g\n", i,
                   expected[i], from_bf16(actual[i]));
      return 1;
    }
  }
  const int current_pages[3] = {0, 0, 2};
  const int current_positions[3] = {0, 1, 2};
  for (int token = 0; token < 3; ++token) {
    for (int d = 0; d < 2; ++d) {
      const size_t index = cache_index(current_pages[token], current_positions[token], d);
      if (key_cache[index] != qkv[token * 8 + 4 + d] ||
          value_cache[index] != qkv[token * 8 + 6 + d]) {
        std::fprintf(stderr, "KV write mismatch token=%d component=%d\n", token, d);
        return 1;
      }
    }
  }
  for (int i = 9; i >= 0; --i) cuMemFree(allocations[i]);
  cuModuleUnload(module);
  cuCtxDestroy(context);
  std::printf("outcome=paged-attention-pass max_abs_error=%g mixed_rows=2 tokens=3\n",
              maximum_error);
  return 0;
}
