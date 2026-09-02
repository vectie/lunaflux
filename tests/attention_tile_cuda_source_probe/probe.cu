#include "generated_attention_tile.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int query_heads = 16;
static constexpr int key_value_heads = 4;
static constexpr int head_dimension = 128;
static constexpr int tokens_per_page = 16;
static constexpr int query_width = query_heads * head_dimension;
static constexpr int input_row_width = 3072;
static constexpr int page_stride_values = 8192;

static void require_cuda(cudaError_t status, const char *step) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(status));
    std::exit(2);
  }
}

template <typename T>
static T *device_copy(const std::vector<T> &host) {
  T *device = nullptr;
  require_cuda(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
  require_cuda(
      cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                 cudaMemcpyHostToDevice),
      "cudaMemcpyHostToDevice");
  return device;
}

static float query_value(int row, int position, int head, int component) {
  return 0.0007f * float(row + 1) + 0.0003f * float(position + 1) +
         0.0001f * float(head + 1) +
         0.0002f * float((component % 11) - 5);
}

static float key_value(int row, int position, int head, int component) {
  return 0.0009f * float(row + 1) + 0.0004f * float(position + 1) +
         0.0002f * float(head + 1) +
         0.0001f * float((component % 7) - 3);
}

static float value_value(int row, int position, int head, int component) {
  return 0.003f * float(row + 1) + 0.001f * float(position + 1) +
         0.0005f * float(head + 1) +
         0.0002f * float((component % 13) - 6);
}

static float run_case(const char *name, const std::vector<int> &lengths) {
  int token_count = 0;
  int page_count = 0;
  for (int length : lengths) {
    token_count += length;
    page_count += (length + tokens_per_page - 1) / tokens_per_page;
  }
  if (lengths.empty() || lengths.size() > 32 || token_count > 128 ||
      page_count > 16) {
    std::fprintf(stderr, "invalid probe case=%s\n", name);
    std::exit(3);
  }

  std::vector<int> counts{int(lengths.size()), 0, int(lengths.size()),
                          token_count, page_count};
  std::vector<int> positions(token_count);
  std::vector<int> row_offsets(lengths.size() + 1, 0);
  std::vector<int> sequence_lengths(lengths);
  std::vector<int> page_offsets(lengths.size() + 1, 0);
  std::vector<int> page_indices(page_count);
  std::vector<int> token_rows(token_count);
  std::vector<__nv_bfloat16> qkv(token_count * input_row_width);
  std::vector<__nv_bfloat16> keys(16 * page_stride_values);
  std::vector<__nv_bfloat16> values(16 * page_stride_values);
  std::vector<__nv_bfloat16> output(token_count * query_width);

  int token_base = 0;
  int page_base = 0;
  for (int row = 0; row < int(lengths.size()); ++row) {
    const int length = lengths[row];
    const int row_pages = (length + tokens_per_page - 1) / tokens_per_page;
    row_offsets[row] = token_base;
    page_offsets[row] = page_base;
    for (int logical_page = 0; logical_page < row_pages; ++logical_page)
      page_indices[page_base + logical_page] = page_base + logical_page;
    for (int position = 0; position < length; ++position) {
      const int token = token_base + position;
      positions[token] = position;
      token_rows[token] = row;
      for (int head = 0; head < query_heads; ++head) {
        for (int component = 0; component < head_dimension; ++component) {
          qkv[token * input_row_width + head * head_dimension + component] =
              __float2bfloat16_rn(
                  query_value(row, position, head, component));
        }
      }
      const int physical_page = page_base + position / tokens_per_page;
      const int token_in_page = position % tokens_per_page;
      for (int head = 0; head < key_value_heads; ++head) {
        for (int component = 0; component < head_dimension; ++component) {
          const int cache_index = physical_page * page_stride_values +
                                  (token_in_page * key_value_heads + head) *
                                      head_dimension +
                                  component;
          keys[cache_index] = __float2bfloat16_rn(
              key_value(row, position, head, component));
          values[cache_index] = __float2bfloat16_rn(
              value_value(row, position, head, component));
        }
      }
    }
    token_base += length;
    page_base += row_pages;
  }
  row_offsets[lengths.size()] = token_base;
  page_offsets[lengths.size()] = page_base;

  int *d_counts = device_copy(counts);
  int *d_positions = device_copy(positions);
  int *d_row_offsets = device_copy(row_offsets);
  int *d_sequence_lengths = device_copy(sequence_lengths);
  int *d_page_offsets = device_copy(page_offsets);
  int *d_page_indices = device_copy(page_indices);
  __nv_bfloat16 *d_qkv = device_copy(qkv);
  __nv_bfloat16 *d_keys = device_copy(keys);
  __nv_bfloat16 *d_values = device_copy(values);
  __nv_bfloat16 *d_output = nullptr;
  require_cuda(cudaMalloc(&d_output, output.size() * sizeof(__nv_bfloat16)),
               "cudaMalloc output");

  require_cuda(
      cudaFuncSetAttribute(lunaflux_attention_prefill_tile_v1,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           LF_SHARED_MEMORY_BYTES),
      "cudaFuncSetAttribute");
  const auto launch = [&]() {
    lunaflux_attention_prefill_tile_v1<<<dim3(128, query_heads),
                                         LF_BLOCK_THREADS,
                                         LF_SHARED_MEMORY_BYTES>>>(
        d_counts, d_positions, d_row_offsets, d_sequence_lengths,
        d_page_offsets, d_page_indices, d_qkv, d_output, d_keys, d_values);
  };
  launch();
  require_cuda(cudaGetLastError(), "kernel launch");
  require_cuda(cudaDeviceSynchronize(), "kernel synchronize");
  require_cuda(cudaMemcpy(output.data(), d_output,
                          output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpyDeviceToHost");

  for (int warmup = 0; warmup < 10; ++warmup) launch();
  require_cuda(cudaDeviceSynchronize(), "benchmark warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  require_cuda(cudaEventCreate(&begin), "cudaEventCreate begin");
  require_cuda(cudaEventCreate(&end), "cudaEventCreate end");
  require_cuda(cudaEventRecord(begin), "cudaEventRecord begin");
  for (int iteration = 0; iteration < 100; ++iteration) launch();
  require_cuda(cudaEventRecord(end), "cudaEventRecord end");
  require_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end");
  float elapsed_millis = 0.0f;
  require_cuda(cudaEventElapsedTime(&elapsed_millis, begin, end),
               "cudaEventElapsedTime");
  require_cuda(cudaEventDestroy(begin), "cudaEventDestroy begin");
  require_cuda(cudaEventDestroy(end), "cudaEventDestroy end");
  std::printf(
      "benchmark_case=%s tokens=%d rows=%zu key_value_tile=%d "
      "block_threads=%d shared_memory_bytes=%lld mean_microseconds=%g\n",
      name, token_count, lengths.size(), LF_KEY_VALUE_TILE, LF_BLOCK_THREADS,
      static_cast<long long>(LF_SHARED_MEMORY_BYTES), elapsed_millis * 10.0f);

  float maximum_absolute_error = 0.0f;
  for (int query = 0; query < token_count; ++query) {
    const int row = token_rows[query];
    const int query_position = positions[query];
    const int row_page_base = page_offsets[row];
    for (int head = 0; head < query_heads; ++head) {
      const int key_value_head = head / (query_heads / key_value_heads);
      std::vector<float> scores(query_position + 1);
      float maximum = -INFINITY;
      for (int key_position = 0; key_position <= query_position;
           ++key_position) {
        const int physical_page =
            page_indices[row_page_base + key_position / tokens_per_page];
        const int token_in_page = key_position % tokens_per_page;
        float dot = 0.0f;
        for (int component = 0; component < head_dimension; ++component) {
          const int cache_index = physical_page * page_stride_values +
                                  (token_in_page * key_value_heads +
                                   key_value_head) *
                                      head_dimension +
                                  component;
          dot += __bfloat162float(
                     qkv[query * input_row_width + head * head_dimension +
                         component]) *
                 __bfloat162float(keys[cache_index]);
        }
        scores[key_position] = dot / std::sqrt(float(head_dimension));
        maximum = std::fmax(maximum, scores[key_position]);
      }
      float denominator = 0.0f;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (int component = 0; component < head_dimension; ++component) {
        float expected = 0.0f;
        for (int key_position = 0; key_position <= query_position;
             ++key_position) {
          const int physical_page =
              page_indices[row_page_base + key_position / tokens_per_page];
          const int token_in_page = key_position % tokens_per_page;
          const int cache_index = physical_page * page_stride_values +
                                  (token_in_page * key_value_heads +
                                   key_value_head) *
                                      head_dimension +
                                  component;
          expected += scores[key_position] / denominator *
                      __bfloat162float(values[cache_index]);
        }
        const float actual = __bfloat162float(
            output[query * query_width + head * head_dimension + component]);
        maximum_absolute_error =
            std::fmax(maximum_absolute_error, std::fabs(actual - expected));
      }
    }
  }

  cudaFree(d_counts);
  cudaFree(d_positions);
  cudaFree(d_row_offsets);
  cudaFree(d_sequence_lengths);
  cudaFree(d_page_offsets);
  cudaFree(d_page_indices);
  cudaFree(d_qkv);
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
  if (!(maximum_absolute_error <= 0.0025f)) {
    std::fprintf(stderr, "case=%s maximum_absolute_error=%g\n", name,
                 maximum_absolute_error);
    std::exit(1);
  }
  std::printf("case=%s tokens=%d rows=%zu maximum_absolute_error=%g\n", name,
              token_count, lengths.size(), maximum_absolute_error);
  return maximum_absolute_error;
}

int main() {
  float maximum_absolute_error = 0.0f;
  maximum_absolute_error =
      std::fmax(maximum_absolute_error, run_case("single-16", {16}));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error, run_case("single-65", {65}));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error, run_case("single-128", {128}));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error, run_case("ragged-17-65", {17, 65}));
  std::printf(
      "outcome=passed token_vectors=16,65,128,17+65 "
      "maximum_absolute_error=%g\n",
      maximum_absolute_error);
  return 0;
}
