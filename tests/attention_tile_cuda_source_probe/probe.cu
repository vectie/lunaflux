#include "generated_attention_tile.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

int main() {
  constexpr int query_heads = 16;
  constexpr int key_value_heads = 4;
  constexpr int head_dimension = 128;
  constexpr int token_count = 16;
  constexpr int query_width = query_heads * head_dimension;
  constexpr int input_row_width = 3072;
  constexpr int page_stride_values = 8192;
  constexpr int dynamic_shared_bytes = 86160;

  std::vector<int> counts{1, 0, 1, token_count, 1};
  std::vector<int> positions(token_count);
  for (int token = 0; token < token_count; ++token) positions[token] = token;
  std::vector<int> row_offsets{0, token_count};
  std::vector<int> sequence_lengths{token_count};
  std::vector<int> page_offsets{0, 1};
  std::vector<int> page_indices{0};
  std::vector<__nv_bfloat16> qkv(token_count * input_row_width);
  std::vector<__nv_bfloat16> keys(page_stride_values);
  std::vector<__nv_bfloat16> values(page_stride_values);
  std::vector<__nv_bfloat16> output(token_count * query_width);

  for (int token = 0; token < token_count; ++token) {
    for (int head = 0; head < query_heads; ++head) {
      for (int component = 0; component < head_dimension; ++component) {
        const float value = 0.002f * float(token + 1) +
                            0.0001f * float(head + 1) +
                            0.0003f * float((component % 11) - 5);
        qkv[token * input_row_width + head * head_dimension + component] =
            __float2bfloat16_rn(value);
      }
    }
    for (int head = 0; head < key_value_heads; ++head) {
      for (int component = 0; component < head_dimension; ++component) {
        const int index =
            (token * key_value_heads + head) * head_dimension + component;
        keys[index] = __float2bfloat16_rn(
            0.0015f * float(token + 1) + 0.0002f * float(head + 1) +
            0.0001f * float((component % 7) - 3));
        values[index] = __float2bfloat16_rn(
            0.01f * float(token + 1) + 0.001f * float(head + 1) +
            0.0002f * float((component % 13) - 6));
      }
    }
  }

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
                           dynamic_shared_bytes),
      "cudaFuncSetAttribute");
  lunaflux_attention_prefill_tile_v1<<<dim3(128, query_heads), 256,
                                       dynamic_shared_bytes>>>(
      d_counts, d_positions, d_row_offsets, d_sequence_lengths, d_page_offsets,
      d_page_indices, d_qkv, d_output, d_keys, d_values);
  require_cuda(cudaGetLastError(), "kernel launch");
  require_cuda(cudaDeviceSynchronize(), "kernel synchronize");
  require_cuda(cudaMemcpy(output.data(), d_output,
                          output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpyDeviceToHost");

  float maximum_absolute_error = 0.0f;
  for (int query = 0; query < token_count; ++query) {
    for (int head = 0; head < query_heads; ++head) {
      const int key_value_head = head / (query_heads / key_value_heads);
      std::vector<float> scores(query + 1);
      float maximum = -INFINITY;
      for (int key = 0; key <= query; ++key) {
        float dot = 0.0f;
        for (int component = 0; component < head_dimension; ++component) {
          dot += __bfloat162float(
                     qkv[query * input_row_width + head * head_dimension +
                         component]) *
                 __bfloat162float(
                     keys[(key * key_value_heads + key_value_head) *
                              head_dimension +
                          component]);
        }
        scores[key] = dot / std::sqrt(float(head_dimension));
        maximum = std::fmax(maximum, scores[key]);
      }
      float denominator = 0.0f;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (int component = 0; component < head_dimension; ++component) {
        float expected = 0.0f;
        for (int key = 0; key <= query; ++key) {
          expected += scores[key] / denominator *
                      __bfloat162float(
                          values[(key * key_value_heads + key_value_head) *
                                     head_dimension +
                                 component]);
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
    std::fprintf(stderr, "maximum_absolute_error=%g\n",
                 maximum_absolute_error);
    return 1;
  }
  std::printf("outcome=passed maximum_absolute_error=%g\n",
              maximum_absolute_error);
  return 0;
}
