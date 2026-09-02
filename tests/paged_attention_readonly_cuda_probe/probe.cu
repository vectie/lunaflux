#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int kQueryHeads = 2;
constexpr int kKvHeads = 1;
constexpr int kHeadDimension = 2;
constexpr int kQueryWidth = kQueryHeads * kHeadDimension;
constexpr int kInputRowWidth = (kQueryHeads + 2 * kKvHeads) * kHeadDimension;
constexpr int kTokensPerPage = 2;
constexpr int kPhysicalPages = 160;
constexpr int kPageStrideValues = 128;
constexpr int kCacheValues = kPhysicalPages * kPageStrideValues;
constexpr int kGuard = 16;
constexpr int kCanaryCells = 260;
constexpr unsigned int kDispatchCanary = 65624u;
constexpr int kCandidateBlockThreads = 128;
constexpr unsigned short kCanary = 0x7e35;
constexpr float kAbsoluteTolerance = 0.0078125f;
constexpr float kRelativeTolerance = 0.01f;

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(e)); return false; } } while (0)
#define CHECK_CU(call) do { CUresult e = (call); if (e != CUDA_SUCCESS) { \
  const char* s = nullptr; cuGetErrorString(e, &s); \
  std::fprintf(stderr, "driver failure %s:%d: %s\n", __FILE__, __LINE__, \
               s ? s : "unknown"); return false; } } while (0)

struct Case {
  const char* name;
  int prefill_rows;
  int decode_rows;
  std::vector<int> query_positions;
  std::vector<int> row_offsets;
  std::vector<int> sequence_lengths;
};

unsigned short bf16_bits(float value) {
  unsigned int bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const unsigned int rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>(rounded >> 16u);
}

float bf16_float(unsigned short value) {
  unsigned int bits = static_cast<unsigned int>(value) << 16u;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

unsigned short hostile_value(int position, int component, int phase) {
  const int raw = (position * (phase + 5) + component * 11 + phase * 17) % 61 - 30;
  return bf16_bits(static_cast<float>(raw) * 0.03125f);
}

int physical_page(int row, int logical_page) {
  return (logical_page * 37 + row * 53 + 11) % kPhysicalPages;
}

__host__ __device__ int cache_index(int page, int position, int component) {
  return page * kPageStrideValues +
         (position % kTokensPerPage) * kHeadDimension + component;
}

__device__ float device_bf16(unsigned short value) {
  return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&value));
}

__device__ unsigned short device_to_bf16(float value) {
  __nv_bfloat16 converted = __float2bfloat16_rn(value);
  return *reinterpret_cast<unsigned short*>(&converted);
}

__global__ void serial_ordered_f32_oracle(
    const int* counts, const int* query_positions, const int* row_offsets,
    const int* sequence_lengths, const int* page_offsets,
    const int* page_indices, const unsigned short* query,
    unsigned short* output, const unsigned short* key_cache,
    const unsigned short* value_cache) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int row_count = counts[2];
  const int token_count = counts[3];
  for (int token = 0; token < token_count; ++token) {
    int row = 0;
    while (row + 1 < row_count && token >= row_offsets[row + 1]) ++row;
    const int position = query_positions[token];
    for (int head = 0; head < kQueryHeads; ++head) {
      float maximum = -CUDART_INF_F;
      for (int key_position = 0; key_position <= position; ++key_position) {
        const int page = page_indices[page_offsets[row] + key_position / kTokensPerPage];
        float dot = 0.0f;
        for (int d = 0; d < kHeadDimension; ++d) {
          dot += device_bf16(query[token * kInputRowWidth + head * kHeadDimension + d]) *
                 device_bf16(key_cache[cache_index(page, key_position, d)]);
        }
        maximum = fmaxf(maximum, dot * rsqrtf(static_cast<float>(kHeadDimension)));
      }
      float denominator = 0.0f;
      for (int key_position = 0; key_position <= position; ++key_position) {
        const int page = page_indices[page_offsets[row] + key_position / kTokensPerPage];
        float dot = 0.0f;
        for (int d = 0; d < kHeadDimension; ++d) {
          dot += device_bf16(query[token * kInputRowWidth + head * kHeadDimension + d]) *
                 device_bf16(key_cache[cache_index(page, key_position, d)]);
        }
        denominator += expf(dot * rsqrtf(static_cast<float>(kHeadDimension)) - maximum);
      }
      for (int d = 0; d < kHeadDimension; ++d) {
        float accumulator = 0.0f;
        for (int key_position = 0; key_position <= position; ++key_position) {
          const int page = page_indices[page_offsets[row] + key_position / kTokensPerPage];
          float dot = 0.0f;
          for (int qd = 0; qd < kHeadDimension; ++qd) {
            dot += device_bf16(query[token * kInputRowWidth + head * kHeadDimension + qd]) *
                   device_bf16(key_cache[cache_index(page, key_position, qd)]);
          }
          accumulator += expf(dot * rsqrtf(static_cast<float>(kHeadDimension)) - maximum) *
                         device_bf16(value_cache[cache_index(page, key_position, d)]) /
                         denominator;
        }
        output[token * kQueryWidth + head * kHeadDimension + d] = device_to_bf16(accumulator);
      }
    }
  }
}

void cpu_ordered_f32_oracle(
    const Case& test, const std::vector<int>& page_offsets,
    const std::vector<int>& page_indices, const std::vector<unsigned short>& query,
    const std::vector<unsigned short>& key, const std::vector<unsigned short>& value,
    std::vector<unsigned short>* output) {
  for (int token = 0; token < static_cast<int>(test.query_positions.size()); ++token) {
    int row = 0;
    while (row + 1 < static_cast<int>(test.sequence_lengths.size()) &&
           token >= test.row_offsets[row + 1]) ++row;
    const int position = test.query_positions[token];
    for (int head = 0; head < kQueryHeads; ++head) {
      std::vector<float> scores(position + 1);
      float maximum = -INFINITY;
      for (int kp = 0; kp <= position; ++kp) {
        const int page = page_indices[page_offsets[row] + kp / kTokensPerPage];
        float dot = 0.0f;
        for (int d = 0; d < kHeadDimension; ++d)
          dot += bf16_float(query[token * kInputRowWidth + head * kHeadDimension + d]) *
                 bf16_float(key[cache_index(page, kp, d)]);
        scores[kp] = dot / std::sqrt(static_cast<float>(kHeadDimension));
        maximum = std::max(maximum, scores[kp]);
      }
      float denominator = 0.0f;
      for (float score : scores) denominator += std::exp(score - maximum);
      for (int d = 0; d < kHeadDimension; ++d) {
        float accumulator = 0.0f;
        for (int kp = 0; kp <= position; ++kp) {
          const int page = page_indices[page_offsets[row] + kp / kTokensPerPage];
          accumulator += std::exp(scores[kp] - maximum) *
                         bf16_float(value[cache_index(page, kp, d)]) / denominator;
        }
        (*output)[token * kQueryWidth + head * kHeadDimension + d] = bf16_bits(accumulator);
      }
    }
  }
}

template <typename T> struct GuardedDevice {
  T* base = nullptr;
  T* data = nullptr;
  size_t count = 0;
  bool allocate(const std::vector<T>& host) {
    count = host.size();
    if (cudaMalloc(&base, (count + 2 * kGuard) * sizeof(T)) != cudaSuccess) return false;
    data = base + kGuard;
    std::vector<T> guarded(count + 2 * kGuard, static_cast<T>(kCanary));
    std::copy(host.begin(), host.end(), guarded.begin() + kGuard);
    return cudaMemcpy(base, guarded.data(), guarded.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool read(std::vector<T>* host) const {
    host->resize(count);
    return cudaMemcpy(host->data(), data, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  bool guards_clean() const {
    std::vector<T> guarded(count + 2 * kGuard);
    if (cudaMemcpy(guarded.data(), base, guarded.size() * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int i = 0; i < kGuard; ++i)
      if (guarded[i] != static_cast<T>(kCanary) || guarded[kGuard + count + i] != static_cast<T>(kCanary)) return false;
    return true;
  }
  void release() { if (base) cudaFree(base); base = nullptr; data = nullptr; }
};

bool run_case(CUfunction function, const Case& test, float* cpu_error,
              float* serial_error, long long* values_checked) {
  const int rows = static_cast<int>(test.sequence_lengths.size());
  const int tokens = static_cast<int>(test.query_positions.size());
  std::vector<int> counts{test.prefill_rows, test.decode_rows, rows, tokens, 0};
  std::vector<int> page_offsets(rows + 1, 0), page_indices;
  std::vector<unsigned short> key(kCacheValues, bf16_bits(-0.5f));
  std::vector<unsigned short> value(kCacheValues, bf16_bits(0.5f));
  for (int row = 0; row < rows; ++row) {
    const int pages = (test.sequence_lengths[row] + kTokensPerPage - 1) / kTokensPerPage;
    for (int logical = 0; logical < pages; ++logical)
      page_indices.push_back(physical_page(row, logical));
    page_offsets[row + 1] = static_cast<int>(page_indices.size());
    for (int position = 0; position < test.sequence_lengths[row]; ++position) {
      const int page = page_indices[page_offsets[row] + position / kTokensPerPage];
      for (int d = 0; d < kHeadDimension; ++d) {
        key[cache_index(page, position, d)] = hostile_value(position + row * 271, d, 1);
        value[cache_index(page, position, d)] = hostile_value(position + row * 271, d, 2);
      }
    }
  }
  counts[4] = static_cast<int>(page_indices.size());
  std::vector<unsigned short> query(tokens * kInputRowWidth, bf16_bits(-7.0f));
  for (int t = 0; t < tokens; ++t)
    for (int d = 0; d < kQueryWidth; ++d)
      query[t * kInputRowWidth + d] = hostile_value(test.query_positions[t], d, 0);
  std::vector<unsigned short> cpu(tokens * kQueryWidth);
  cpu_ordered_f32_oracle(test, page_offsets, page_indices, query, key, value, &cpu);

  GuardedDevice<int> d_counts, d_positions, d_rows, d_lengths, d_page_offsets, d_pages;
  GuardedDevice<unsigned short> d_query, d_output, d_serial, d_key, d_value;
  GuardedDevice<unsigned int> d_canary;
  std::vector<unsigned short> output(tokens * kQueryWidth, bf16_bits(123.0f));
  std::vector<unsigned int> canary(kCanaryCells, 0u);
  bool ok = d_counts.allocate(counts) && d_positions.allocate(test.query_positions) &&
    d_rows.allocate(test.row_offsets) && d_lengths.allocate(test.sequence_lengths) &&
    d_page_offsets.allocate(page_offsets) && d_pages.allocate(page_indices) &&
    d_query.allocate(query) && d_output.allocate(output) && d_serial.allocate(output) &&
    d_key.allocate(key) && d_value.allocate(value) && d_canary.allocate(canary);
  if (!ok) return false;
  std::vector<unsigned short> key_before, value_before;
  if (!d_key.read(&key_before) || !d_value.read(&value_before)) return false;
  void* args[] = {&d_counts.data, &d_positions.data, &d_rows.data, &d_lengths.data,
    &d_page_offsets.data, &d_pages.data, &d_query.data, &d_output.data,
    &d_key.data, &d_value.data, &d_canary.data};
  const unsigned int grid_x = std::min(tokens, 128);
  CHECK_CU(cuLaunchKernel(function, grid_x, kQueryHeads, 1,
                          kCandidateBlockThreads, 1, 1, 0,
                          nullptr, args, nullptr));
  serial_ordered_f32_oracle<<<1, 1>>>(d_counts.data, d_positions.data, d_rows.data,
    d_lengths.data, d_page_offsets.data, d_pages.data, d_query.data, d_serial.data,
    d_key.data, d_value.data);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<unsigned short> observed, serial, key_after, value_after;
  std::vector<unsigned int> canary_after;
  if (!d_output.read(&observed) || !d_serial.read(&serial) ||
      !d_key.read(&key_after) || !d_value.read(&value_after) ||
      !d_canary.read(&canary_after)) return false;
  if (key_before != key_after || value_before != value_after) return false;
  if (!(d_counts.guards_clean() && d_positions.guards_clean() && d_rows.guards_clean() &&
      d_lengths.guards_clean() && d_page_offsets.guards_clean() && d_pages.guards_clean() &&
      d_query.guards_clean() && d_output.guards_clean() && d_serial.guards_clean() &&
      d_key.guards_clean() && d_value.guards_clean() && d_canary.guards_clean())) return false;
  unsigned long long canary_sum = 0;
  for (int i = 0; i < kCanaryCells; ++i) {
    const unsigned int expected = i < tokens ? kDispatchCanary : 0u;
    if (canary_after[i] != expected) return false;
    canary_sum += canary_after[i];
  }
  if (canary_sum != static_cast<unsigned long long>(tokens) * kDispatchCanary) return false;
  for (size_t i = 0; i < observed.size(); ++i) {
    const float got = bf16_float(observed[i]);
    const float cpu_value = bf16_float(cpu[i]);
    const float serial_value = bf16_float(serial[i]);
    *cpu_error = std::max(*cpu_error, std::fabs(got - cpu_value));
    *serial_error = std::max(*serial_error, std::fabs(got - serial_value));
    const float allowed = kAbsoluteTolerance + kRelativeTolerance * std::fabs(cpu_value);
    const float serial_allowed =
      kAbsoluteTolerance + kRelativeTolerance * std::fabs(serial_value);
    if (!std::isfinite(got) || std::fabs(got - cpu_value) > allowed ||
        std::fabs(got - serial_value) > serial_allowed) return false;
    ++*values_checked;
  }
  d_counts.release(); d_positions.release(); d_rows.release(); d_lengths.release();
  d_page_offsets.release(); d_pages.release(); d_query.release(); d_output.release();
  d_serial.release(); d_key.release(); d_value.release();
  d_canary.release();
  return true;
}

std::string hex_name(const char* name) {
  static const char* digits = "0123456789abcdef";
  std::string result;
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(name); *p; ++p) {
    result.push_back(digits[*p >> 4]); result.push_back(digits[*p & 15]);
  }
  return result;
}

std::string device_uuid(const cudaUUID_t& uuid) {
  static const char* digits = "0123456789abcdef";
  std::string result = "GPU-";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result.push_back('-');
    const unsigned char value = static_cast<unsigned char>(uuid.bytes[i]);
    result.push_back(digits[value >> 4]); result.push_back(digits[value & 15]);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const int cycles = std::atoi(argv[3]);
  if (cycles <= 0 || cycles > 32) return 2;
  if (cuInit(0) != CUDA_SUCCESS || cudaSetDevice(0) != cudaSuccess) return 1;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      properties.major != 12 || properties.minor != 0) return 1;
  CUmodule module = nullptr; CUfunction function = nullptr;
  if (cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
      cuModuleGetFunction(&function, module, argv[2]) != CUDA_SUCCESS) return 1;
  const std::vector<Case> cases{
    {"origin", 1, 0, {0}, {0, 1}, {1}},
    {"page-tail", 0, 1, {1}, {0, 1}, {2}},
    {"cross-page", 1, 0, {1, 2}, {0, 2}, {3}},
    {"multirow", 1, 1, {2, 4}, {0, 1, 2}, {3, 5}},
    {"long-context", 1, 0, {254, 255, 256}, {0, 3}, {257}},
  };
  float cpu_error = 0.0f, serial_error = 0.0f;
  long long values_checked = 0;
  for (int cycle = 0; cycle < cycles; ++cycle)
    for (const Case& test : cases)
      if (!run_case(function, test, &cpu_error, &serial_error, &values_checked)) return 1;
  if (cuModuleUnload(module) != CUDA_SUCCESS || cudaDeviceReset() != cudaSuccess) return 1;
  char pci_bus_id[32]{};
  std::snprintf(pci_bus_id, sizeof(pci_bus_id), "%08x:%02x:%02x.0",
    properties.pciDomainID, properties.pciBusID, properties.pciDeviceID);
  std::printf(
    "outcome=paged-attention-readonly-sm120-qualification-pass cycles=%d case_families=origin,page-tail,cross-page,multirow,long-context scheduler_modes=prefill-only,decode-only,mixed-prefill-decode numeric_case_count=%lld candidate_launches=%d serial_launches=%d cpu_vs_candidate_max_abs_error=%.9g serial_vs_candidate_max_abs_error=%.9g absolute_tolerance=0.0078125 relative_tolerance=0.01 output_dtype=bf16 cpu_oracle=independent-ordered-f32-v1 serial_cuda_oracle=independent-ordered-f32-kernel-v1 cache_snapshot_bytes=%d cache_snapshot_unchanged=true input_guards_unchanged=true output_guards_unchanged=true dispatch_symbol_resolved=true dispatch_grid_x_max=128 dispatch_grid_y=2 dispatch_block_x=128 dispatch_canary_per_token=65624 dispatch_canary_cell_count=260 dispatch_canary_exact=true dispatch_canary_tail_zero=true dispatch_canary_sum_checked=true input_row_width=8 target=sm_120 device_ordinal=0 device_uuid=%s device_pci_bus_id=%s device_name_hex=%s device_total_memory_bytes=%zu resources=module0,allocation0,device_bytes0,pending0,cleanup0 manifest_bindable=false promotion_authority=absent\n",
    cycles, values_checked, cycles * 5, cycles * 5, cpu_error, serial_error,
    2 * kCacheValues * static_cast<int>(sizeof(unsigned short)),
    device_uuid(properties.uuid).c_str(), pci_bus_id,
    hex_name(properties.name).c_str(), properties.totalGlobalMem);
  return 0;
}
