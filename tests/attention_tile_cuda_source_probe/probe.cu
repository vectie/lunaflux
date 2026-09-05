#include "generated_attention_tile.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int query_heads = 16;
static constexpr int key_value_heads = 4;
static constexpr int head_dimension = LF_HEAD_DIMENSION;
static constexpr int tokens_per_page = 16;
static constexpr int query_width = query_heads * head_dimension;
static constexpr int input_row_width = (query_heads + 2 * key_value_heads) * head_dimension;
static constexpr int page_stride_values = tokens_per_page * key_value_heads * head_dimension;

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

static float run_case(const char *name, const std::vector<int> &query_lengths,
                      const std::vector<int> &context_lengths,
                      bool exhaustive_reference) {
  if (query_lengths.size() != context_lengths.size()) {
    std::fprintf(stderr, "mismatched probe case=%s\n", name);
    std::exit(3);
  }
  int token_count = 0;
  int page_count = 0;
  int maximum_context_tokens = 0;
  int query_tile_count = 0;
  for (int row = 0; row < int(query_lengths.size()); ++row) {
    token_count += query_lengths[row];
    page_count +=
        (context_lengths[row] + tokens_per_page - 1) / tokens_per_page;
    maximum_context_tokens =
        std::max(maximum_context_tokens, context_lengths[row]);
    query_tile_count +=
        (query_lengths[row] + LF_QUERY_TILE - 1) / LF_QUERY_TILE;
  }
  if (query_lengths.empty() || query_lengths.size() > 32 ||
      token_count > 128 || page_count > 256 || query_tile_count <= 0) {
    std::fprintf(stderr, "invalid probe case=%s\n", name);
    std::exit(3);
  }
  for (int row = 0; row < int(query_lengths.size()); ++row) {
    if (query_lengths[row] <= 0 ||
        context_lengths[row] < query_lengths[row] ||
        context_lengths[row] > 4096) {
      std::fprintf(stderr, "invalid probe row case=%s row=%d\n", name, row);
      std::exit(3);
    }
  }

  std::vector<int> counts{int(query_lengths.size()), 0,
                          int(query_lengths.size()),
                          token_count, page_count};
  std::vector<int> positions(token_count);
  std::vector<int> row_offsets(query_lengths.size() + 1, 0);
  std::vector<int> sequence_lengths(context_lengths);
  std::vector<int> page_offsets(query_lengths.size() + 1, 0);
  std::vector<int> page_indices(page_count);
  std::vector<int> token_rows(token_count);
  std::vector<__nv_bfloat16> qkv(token_count * input_row_width);
  std::vector<__nv_bfloat16> keys(256 * page_stride_values);
  std::vector<__nv_bfloat16> values(256 * page_stride_values);
  std::vector<__nv_bfloat16> output(token_count * query_width);

  int token_base = 0;
  int page_base = 0;
  for (int row = 0; row < int(query_lengths.size()); ++row) {
    const int query_length = query_lengths[row];
    const int context_length = context_lengths[row];
    const int query_position_base = context_length - query_length;
    const int row_pages =
        (context_length + tokens_per_page - 1) / tokens_per_page;
    row_offsets[row] = token_base;
    page_offsets[row] = page_base;
    for (int logical_page = 0; logical_page < row_pages; ++logical_page)
      page_indices[page_base + logical_page] = page_base + logical_page;
    for (int query = 0; query < query_length; ++query) {
      const int position = query_position_base + query;
      const int token = token_base + query;
      positions[token] = position;
      token_rows[token] = row;
      for (int head = 0; head < query_heads; ++head) {
        for (int component = 0; component < head_dimension; ++component) {
          qkv[token * input_row_width + head * head_dimension + component] =
              __float2bfloat16_rn(
                  query_value(row, position, head, component));
        }
      }
    }
    for (int position = 0; position < context_length; ++position) {
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
    token_base += query_length;
    page_base += row_pages;
  }
  row_offsets[query_lengths.size()] = token_base;
  page_offsets[query_lengths.size()] = page_base;

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

#ifdef LF_PARTITIONED_PROBE
  float *d_workspace = nullptr;
  constexpr unsigned long long workspace_values =
      LF_PARTIAL_STATE_COUNT * (2ULL + LF_HEAD_DIMENSION);
  require_cuda(cudaMalloc(&d_workspace, workspace_values * sizeof(float)),
               "cudaMalloc partitioned workspace");
  require_cuda(
      cudaFuncSetAttribute(lunaflux_attention_prefill_partitioned_partial_v1,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           LF_SHARED_MEMORY_BYTES),
      "cudaFuncSetAttribute partitioned partial");
  const auto launch = [&]() {
    lunaflux_attention_prefill_partitioned_partial_v1
        <<<dim3(query_tile_count, query_heads, LF_KEY_VALUE_PARTITIONS),
           LF_BLOCK_THREADS,
           LF_SHARED_MEMORY_BYTES>>>(
            d_counts, d_positions, d_row_offsets, d_sequence_lengths,
            d_page_offsets, d_page_indices, d_qkv, d_keys, d_values,
            d_workspace);
    lunaflux_attention_prefill_partitioned_merge_v1
        <<<dim3(query_tile_count, query_heads), LF_MERGE_BLOCK_THREADS>>>(
            d_counts, d_row_offsets, d_workspace, d_output);
  };
#else
  require_cuda(
      cudaFuncSetAttribute(lunaflux_attention_prefill_tile_v1,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           LF_SHARED_MEMORY_BYTES),
      "cudaFuncSetAttribute");
  const auto launch = [&]() {
    lunaflux_attention_prefill_tile_v1<<<dim3(query_tile_count, query_heads),
                                         LF_BLOCK_THREADS,
                                         LF_SHARED_MEMORY_BYTES>>>(
        d_counts, d_positions, d_row_offsets, d_sequence_lengths,
        d_page_offsets, d_page_indices, d_qkv, d_output, d_keys, d_values);
  };
#endif
  launch();
  require_cuda(cudaGetLastError(), "kernel launch");
  require_cuda(cudaDeviceSynchronize(), "kernel synchronize");
  require_cuda(cudaMemcpy(output.data(), d_output,
                          output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpyDeviceToHost");

#ifndef LF_SKIP_BENCHMARK
  for (int warmup = 0; warmup < 10; ++warmup) launch();
  require_cuda(cudaDeviceSynchronize(), "benchmark warmup");
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  require_cuda(cudaEventCreate(&begin), "cudaEventCreate begin");
  require_cuda(cudaEventCreate(&end), "cudaEventCreate end");
  std::vector<float> latency_samples;
  for (int sample = 0; sample < 9; ++sample) {
    require_cuda(cudaEventRecord(begin), "cudaEventRecord begin");
    for (int iteration = 0; iteration < 40; ++iteration) launch();
    require_cuda(cudaEventRecord(end), "cudaEventRecord end");
    require_cuda(cudaEventSynchronize(end), "cudaEventSynchronize end");
    float elapsed_millis = 0.0f;
    require_cuda(cudaEventElapsedTime(&elapsed_millis, begin, end),
                 "cudaEventElapsedTime");
    latency_samples.push_back(elapsed_millis * 25.0f);
  }
  require_cuda(cudaEventDestroy(begin), "cudaEventDestroy begin");
  require_cuda(cudaEventDestroy(end), "cudaEventDestroy end");
  std::sort(latency_samples.begin(), latency_samples.end());
  const float median_microseconds = latency_samples[latency_samples.size() / 2];
  std::printf(
      "benchmark_case=%s query_tokens=%d context_tokens=%d rows=%zu "
      "query_tiles=%d key_value_tile=%d block_threads=%d "
      "shared_memory_bytes=%lld median_microseconds=%g\n",
      name, token_count, maximum_context_tokens, query_lengths.size(),
      query_tile_count, LF_KEY_VALUE_TILE, LF_BLOCK_THREADS,
      static_cast<long long>(LF_SHARED_MEMORY_BYTES), median_microseconds);
#endif

  float maximum_absolute_error = 0.0f;
  float maximum_normalized_error = 0.0f;
  int maximum_error_query = -1;
  int maximum_error_head = -1;
  int maximum_error_component = -1;
  float maximum_error_expected = 0.0f;
  float maximum_error_actual = 0.0f;
  for (int query = 0; query < token_count; ++query) {
    const int row = token_rows[query];
    const int query_position = positions[query];
    const int row_page_base = page_offsets[row];
    for (int head = 0; head < query_heads; ++head) {
      const bool sample_query = query == row_offsets[row] ||
                                query + 1 == row_offsets[row + 1] ||
                                query == (row_offsets[row] +
                                          row_offsets[row + 1]) /
                                             2;
      const bool sample_head = head == 0 || head + 1 == query_heads;
      if (!exhaustive_reference && !(sample_query && sample_head)) continue;
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
        const bool sample_component = component == 0 || component == 17 ||
                                      component + 1 == head_dimension;
        if (!exhaustive_reference && !sample_component) continue;
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
        const float absolute_error = std::fabs(actual - expected);
        const float allowed_error = 0.0025f + 0.005f * std::fabs(expected);
        maximum_normalized_error =
            std::fmax(maximum_normalized_error, absolute_error / allowed_error);
        if (absolute_error > maximum_absolute_error) {
          maximum_absolute_error = absolute_error;
          maximum_error_query = query;
          maximum_error_head = head;
          maximum_error_component = component;
          maximum_error_expected = expected;
          maximum_error_actual = actual;
        }
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
#ifdef LF_PARTITIONED_PROBE
  cudaFree(d_workspace);
#endif
  if (!(maximum_normalized_error <= 1.0f)) {
    std::fprintf(
        stderr,
        "case=%s maximum_absolute_error=%g maximum_normalized_error=%g "
        "query=%d head=%d component=%d expected=%g actual=%g\n",
        name, maximum_absolute_error, maximum_normalized_error,
        maximum_error_query, maximum_error_head, maximum_error_component,
        maximum_error_expected, maximum_error_actual);
    std::exit(1);
  }
  std::printf(
      "case=%s query_tokens=%d context_tokens=%d rows=%zu "
      "maximum_absolute_error=%g maximum_normalized_error=%g reference=%s\n",
      name, token_count, maximum_context_tokens, query_lengths.size(),
      maximum_absolute_error, maximum_normalized_error,
      exhaustive_reference ? "exhaustive" : "deterministic-sample");
  return maximum_absolute_error;
}

int main() {
  float maximum_absolute_error = 0.0f;
  maximum_absolute_error =
      std::fmax(maximum_absolute_error,
                run_case("single-16", {16}, {16}, true));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error,
                run_case("single-65", {65}, {65}, true));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error,
                run_case("single-128", {128}, {128}, true));
  maximum_absolute_error =
      std::fmax(maximum_absolute_error,
                run_case("ragged-17-65", {17, 65}, {17, 65}, true));
  for (int query_tokens : {16, 64, 128}) {
    for (int context_tokens : {512, 1024, 2048, 4096}) {
      char name[64];
      std::snprintf(name, sizeof(name), "q%d-context-%d", query_tokens,
                    context_tokens);
      maximum_absolute_error = std::fmax(
          maximum_absolute_error,
          run_case(name, {query_tokens}, {context_tokens}, false));
    }
  }
  std::printf(
      "outcome=passed query_token_vectors=16,64,128 "
      "context_token_vectors=16,65,128,512,1024,2048,4096 "
      "maximum_absolute_error=%g\n",
      maximum_absolute_error);
  return 0;
}
