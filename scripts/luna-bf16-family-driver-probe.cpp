#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_ok(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) return;
  const char *name = "CUDA_ERROR_UNKNOWN";
  const char *message = "unknown CUDA driver failure";
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) + ": " + name + ": " + message);
}

uint16_t bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + rounding) >> 16U);
}

float f32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float ordered_mul(float left, float right) {
  volatile float result = left * right;
  return result;
}

float ordered_add(float left, float right) {
  volatile float result = left + right;
  return result;
}

struct Buffer {
  CUdeviceptr pointer = 0;
  size_t bytes = 0;

  explicit Buffer(size_t size) : bytes(size) {
    cuda_ok(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
  }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  ~Buffer() {
    if (pointer != 0) (void)cuMemFree(pointer);
  }

  void close() {
    if (pointer == 0) return;
    cuda_ok(cuMemFree(pointer), "cuMemFree");
    pointer = 0;
  }

  template <typename T> void upload(const std::vector<T> &values) {
    if (values.size() * sizeof(T) != bytes) throw std::runtime_error("upload size mismatch");
    cuda_ok(cuMemcpyHtoD(pointer, values.data(), bytes), "cuMemcpyHtoD");
  }

  template <typename T> std::vector<T> download() const {
    if (bytes % sizeof(T) != 0) throw std::runtime_error("download size mismatch");
    std::vector<T> result(bytes / sizeof(T));
    cuda_ok(cuMemcpyDtoH(result.data(), pointer, bytes), "cuMemcpyDtoH");
    return result;
  }
};

struct Kernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;

  Kernel(const std::string &path, const char *symbol) {
    cuda_ok(cuModuleLoad(&module, path.c_str()), "cuModuleLoad");
    try {
      cuda_ok(cuModuleGetFunction(&function, module, symbol), "cuModuleGetFunction");
    } catch (...) {
      (void)cuModuleUnload(module);
      module = nullptr;
      throw;
    }
  }

  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;

  ~Kernel() {
    if (module != nullptr) (void)cuModuleUnload(module);
  }

  void close() {
    if (module == nullptr) return;
    cuda_ok(cuModuleUnload(module), "cuModuleUnload");
    module = nullptr;
    function = nullptr;
  }

  void launch(unsigned grid_x, void **arguments) const {
    cuda_ok(cuLaunchKernel(function, grid_x, 1, 1, 256, 1, 1, 0, nullptr, arguments, nullptr),
            "cuLaunchKernel");
    cuda_ok(cuCtxSynchronize(), "cuCtxSynchronize");
  }
};

std::vector<uint16_t> encoded(const std::vector<float> &values) {
  std::vector<uint16_t> result;
  result.reserve(values.size());
  for (float value : values) result.push_back(bf16(value));
  return result;
}

void compare(const char *family, const std::vector<uint16_t> &actual,
             const std::vector<float> &expected, size_t live, float atol, float rtol) {
  if (actual.size() != expected.size()) throw std::runtime_error(std::string(family) + " size mismatch");
  for (size_t index = 0; index < live; ++index) {
    const float got = f32(actual[index]);
    const float want = f32(bf16(expected[index]));
    const float tolerance = atol + rtol * std::fabs(want);
    if (!std::isfinite(got) || std::fabs(got - want) > tolerance) {
      throw std::runtime_error(std::string(family) + " mismatch at " + std::to_string(index));
    }
  }
  for (size_t index = live; index < actual.size(); ++index) {
    if (actual[index] != bf16(91.0F)) {
      throw std::runtime_error(std::string(family) + " wrote beyond live rows");
    }
  }
}

std::vector<float> base_input() {
  return {0.0F, 1.0F, -1.0F, 0.5F, 4.0F, -3.0F, 0.25F, -0.125F,
          -2.0F, 2.0F, 8.0F, -8.0F, 9.0F, 9.0F, 9.0F, 9.0F};
}

std::vector<float> weights(size_t rows, size_t columns, float scale, int phase) {
  std::vector<float> result(rows * columns);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      const int signed_value = static_cast<int>((row * 7 + column * 3 + phase) % 11) - 5;
      result[row * columns + column] = static_cast<float>(signed_value) * scale;
    }
  }
  return result;
}

void projection_reference(const std::vector<uint16_t> &input,
                          const std::vector<uint16_t> &weight, size_t rows,
                          size_t input_width, size_t output_width,
                          std::vector<float> *output) {
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < output_width; ++column) {
      float accumulator = 0.0F;
      for (size_t inner = 0; inner < input_width; ++inner) {
        const float product = ordered_mul(
            f32(input[row * input_width + inner]),
            f32(weight[column * input_width + inner]));
        accumulator = ordered_add(accumulator, product);
      }
      (*output)[row * output_width + column] = accumulator;
    }
  }
}

void test_embedding(const std::string &module_path, Buffer &counts) {
  const std::vector<int32_t> tokens = {15, 0, -1, 7};
  std::vector<float> table(16 * 8);
  for (size_t row = 0; row < 16; ++row)
    for (size_t column = 0; column < 8; ++column)
      table[row * 8 + column] = (static_cast<float>(row) - 8.0F) * 0.5F +
                                static_cast<float>(column) * 0.0625F;
  const auto table_bits = encoded(table);
  std::vector<uint16_t> output(32, bf16(91.0F));
  Buffer token_device(tokens.size() * sizeof(int32_t));
  Buffer table_device(table_bits.size() * sizeof(uint16_t));
  Buffer output_device(output.size() * sizeof(uint16_t));
  token_device.upload(tokens); table_device.upload(table_bits); output_device.upload(output);
  Kernel kernel(module_path, "lunaflux_luna_embedding_lookup_v1_op_0_ep_1");
  void *args[] = {&counts.pointer, &token_device.pointer, &table_device.pointer, &output_device.pointer};
  kernel.launch(1, args);
  const auto actual = output_device.download<uint16_t>();
  std::vector<float> expected(32, 91.0F);
  for (size_t column = 0; column < 8; ++column) {
    expected[column] = f32(table_bits[15 * 8 + column]);
    expected[8 + column] = f32(table_bits[column]);
    expected[16 + column] = 0.0F;
  }
  compare("embedding", actual, expected, 24, 0.0F, 0.0F);
  kernel.close(); output_device.close(); table_device.close(); token_device.close();
}

void test_rms(const std::string &module_path, Buffer &counts) {
  const auto input = encoded({0,0,0,0,0,0,0,0, 1,-1,2,-2,4,-4,0.25F,-0.25F,
                              32,-32,0.0078125F,-0.0078125F,3,-5,7,-9, 1,1,1,1,1,1,1,1});
  const auto weight = encoded({1,0.5F,-1,2,0.25F,-0.5F,1.5F,-2});
  std::vector<uint16_t> output(32, bf16(91.0F));
  Buffer in_device(input.size() * 2);
  Buffer weight_device(weight.size() * 2);
  Buffer out_device(output.size() * 2);
  in_device.upload(input);
  weight_device.upload(weight);
  out_device.upload(output);
  Kernel kernel(module_path, "lunaflux_luna_rms_norm_v1_op_1_ep_2");
  void *args[] = {&counts.pointer, &in_device.pointer, &weight_device.pointer, &out_device.pointer};
  kernel.launch(4, args);
  std::vector<float> expected(32, 91.0F);
  for (size_t row = 0; row < 3; ++row) {
    float sum = 0.0F;
    for (size_t column = 0; column < 8; ++column) {
      const float value = f32(input[row * 8 + column]);
      sum = ordered_add(sum, ordered_mul(value, value));
    }
    const float inverse = 1.0F / std::sqrt(sum / 8.0F + 0.00001F);
    for (size_t column = 0; column < 8; ++column)
      expected[row * 8 + column] = ordered_mul(
          ordered_mul(f32(input[row * 8 + column]), inverse),
          f32(weight[column]));
  }
  compare("rms_norm", out_device.download<uint16_t>(), expected, 24, 0.004F, 0.004F);
  kernel.close();
  out_device.close();
  weight_device.close();
  in_device.close();
}

void test_residual(const std::string &module_path, Buffer &counts) {
  const auto left = encoded({1,-1,0.0078125F,-0.0078125F,32,-32,3,-5,
                             8,-8,0,0.5F,-0.5F,2,-2,16,
                             -7,7,4,-4,0.25F,-0.25F,64,-64, 1,1,1,1,1,1,1,1});
  std::vector<float> right_values(32, 0.0F);
  for (size_t index = 0; index < 24; ++index) right_values[index] = -0.75F * f32(left[index]);
  const auto right = encoded(right_values);
  std::vector<uint16_t> output(32, bf16(91.0F));
  Buffer left_device(left.size() * 2);
  Buffer right_device(right.size() * 2);
  Buffer out_device(output.size() * 2);
  left_device.upload(left);
  right_device.upload(right);
  out_device.upload(output);
  Kernel kernel(module_path, "lunaflux_luna_residual_add_v1_op_2_ep_3");
  void *args[] = {&counts.pointer, &left_device.pointer, &right_device.pointer, &out_device.pointer};
  kernel.launch(1, args);
  std::vector<float> expected(32, 91.0F);
  for (size_t index = 0; index < 24; ++index) expected[index] = f32(left[index]) + f32(right[index]);
  compare("residual", out_device.download<uint16_t>(), expected, 24, 0.0F, 0.0F);
  kernel.close();
  out_device.close();
  right_device.close();
  left_device.close();
}

void test_rope(const std::string &module_path, Buffer &counts) {
  std::vector<float> input_values(64);
  for (size_t index = 0; index < input_values.size(); ++index)
    input_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.25F;
  const auto input = encoded(input_values);
  const std::vector<int32_t> positions = {0, 1, 7, 31};
  std::vector<uint16_t> output(64, bf16(91.0F));
  Buffer pos_device(positions.size() * 4);
  Buffer in_device(input.size() * 2);
  Buffer out_device(output.size() * 2);
  pos_device.upload(positions);
  in_device.upload(input);
  out_device.upload(output);
  Kernel kernel(module_path, "lunaflux_luna_positioned_rotary_v1_op_3_ep_4");
  void *args[] = {&counts.pointer, &pos_device.pointer, &in_device.pointer, &out_device.pointer};
  kernel.launch(1, args);
  std::vector<float> expected(64, 91.0F);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t column = 0; column < 16; ++column) {
      if (column >= 12) {
        expected[row * 16 + column] = f32(input[row * 16 + column]);
        continue;
      }
      const size_t lane = column % 4;
      const size_t pair = lane % 2;
      const size_t head = column - lane;
      const float exponent = static_cast<float>(pair * 2) / 4.0F;
      const float angle = static_cast<float>(positions[row]) /
                          std::pow(10000.0F, exponent);
      const float first = f32(input[row * 16 + head + pair]);
      const float second = f32(input[row * 16 + head + pair + 2]);
      expected[row * 16 + column] =
          lane < 2 ? first * std::cos(angle) - second * std::sin(angle)
                   : second * std::cos(angle) + first * std::sin(angle);
    }
  }
  compare("rope", out_device.download<uint16_t>(), expected, 48, 0.004F, 0.004F);
  kernel.close();
  out_device.close();
  in_device.close();
  pos_device.close();
}

void test_qkv(const std::string &module_path, Buffer &counts) {
  const auto input = encoded(base_input());
  const auto q = encoded(weights(4, 4, 0.125F, 1));
  const auto k = encoded(weights(2, 4, 0.25F, 2));
  const auto v = encoded(weights(2, 4, 0.0625F, 3));
  std::vector<uint16_t> output(32, bf16(91.0F));
  Buffer in(input.size() * 2), qd(q.size() * 2), kd(k.size() * 2);
  Buffer vd(v.size() * 2), out(output.size() * 2);
  in.upload(input);
  qd.upload(q);
  kd.upload(k);
  vd.upload(v);
  out.upload(output);
  Kernel kernel(module_path, "lunaflux_luna_qkv_projection_bf16_reference_v1_op_1_ep_2");
  void *args[] = {&counts.pointer, &in.pointer, &qd.pointer,
                  &kd.pointer, &vd.pointer, &out.pointer};
  kernel.launch(1, args);
  std::vector<float> expected(32, 91.0F), q_expected(12), k_expected(6),
      v_expected(6);
  projection_reference(input, q, 3, 4, 4, &q_expected);
  projection_reference(input, k, 3, 4, 2, &k_expected);
  projection_reference(input, v, 3, 4, 2, &v_expected);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t column = 0; column < 4; ++column)
      expected[row * 8 + column] = q_expected[row * 4 + column];
    for (size_t column = 0; column < 2; ++column) {
      expected[row * 8 + 4 + column] = k_expected[row * 2 + column];
      expected[row * 8 + 6 + column] = v_expected[row * 2 + column];
    }
  }
  compare("qkv", out.download<uint16_t>(), expected, 24, 0, 0);
  kernel.close(); out.close(); vd.close(); kd.close(); qd.close(); in.close();
}

void test_dense_family(const std::string &module_path, const char *symbol,
                       size_t output_width, int phase, const char *family) {
  const auto input = encoded(base_input());
  const auto weight = encoded(weights(output_width, 4, 0.125F, phase));
  std::vector<uint16_t> output(4 * output_width, bf16(91.0F));
  Buffer counts(20), in(input.size() * 2), wd(weight.size() * 2);
  Buffer out(output.size() * 2);
  const std::vector<int32_t> count_values = {0, 0, 0, 3, 0};
  counts.upload(count_values);
  in.upload(input);
  wd.upload(weight);
  out.upload(output);
  Kernel kernel(module_path, symbol);
  void *args[] = {&counts.pointer, &in.pointer, &wd.pointer, &out.pointer};
  kernel.launch(1, args);
  std::vector<float> expected(4 * output_width, 91.0F);
  projection_reference(input, weight, 3, 4, output_width, &expected);
  compare(family, out.download<uint16_t>(), expected, 3 * output_width, 0, 0);
  kernel.close(); out.close(); wd.close(); in.close(); counts.close();
}

void test_mlp(const std::string &module_path, Buffer &counts) {
  const auto input = encoded(base_input());
  const auto gate = encoded(weights(8, 4, 0.125F, 4));
  const auto up = encoded(weights(8, 4, 0.0625F, 5));
  const auto down = encoded(weights(4, 8, 0.125F, 6));
  std::vector<uint16_t> output(16, bf16(91.0F));
  Buffer in(input.size() * 2), gd(gate.size() * 2), ud(up.size() * 2);
  Buffer dd(down.size() * 2), out(output.size() * 2);
  in.upload(input); gd.upload(gate); ud.upload(up); dd.upload(down); out.upload(output);
  Kernel kernel(module_path,
                "lunaflux_luna_gated_mlp_bf16_reference_v1_op_5_ep_6");
  void *args[] = {&counts.pointer, &in.pointer, &gd.pointer,
                  &ud.pointer, &dd.pointer, &out.pointer};
  kernel.launch(1, args);
  std::vector<float> expected(16, 91.0F);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t column = 0; column < 4; ++column) {
      float accumulator = 0.0F;
      for (size_t middle = 0; middle < 8; ++middle) {
        float gate_value = 0.0F, up_value = 0.0F;
        for (size_t inner = 0; inner < 4; ++inner) {
          const float input_value = f32(input[row * 4 + inner]);
          gate_value = ordered_add(
              gate_value, ordered_mul(input_value, f32(gate[middle * 4 + inner])));
          up_value = ordered_add(
              up_value, ordered_mul(input_value, f32(up[middle * 4 + inner])));
        }
        const float silu = gate_value /
                           ordered_add(1.0F, std::exp(-gate_value));
        const float combined = ordered_mul(silu, up_value);
        accumulator = ordered_add(
            accumulator, ordered_mul(combined, f32(down[column * 8 + middle])));
      }
      expected[row * 4 + column] = accumulator;
    }
  }
  compare("mlp", out.download<uint16_t>(), expected, 12, 0.008F, 0.008F);
  kernel.close(); out.close(); dd.close(); ud.close(); gd.close(); in.close();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cerr << "usage: probe EMBEDDING RMS RESIDUAL ROPE QKV DENSE MLP LM_HEAD\n";
    return 2;
  }
  CUcontext context = nullptr;
  try {
    cuda_ok(cuInit(0), "cuInit");
    CUdevice device = 0; cuda_ok(cuDeviceGet(&device, 0), "cuDeviceGet");
    int major = 0, minor = 0;
    cuda_ok(cuDeviceGetAttribute(
                &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
            "cuDeviceGetAttribute(major)");
    cuda_ok(cuDeviceGetAttribute(
                &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
            "cuDeviceGetAttribute(minor)");
    if (major != 12 || minor != 0)
      throw std::runtime_error("physical fixture requires exact sm120 device 0");
    cuda_ok(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    cuda_ok(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    const std::vector<int32_t> count_values = {0, 0, 0, 3, 0};
    Buffer counts(20);
    counts.upload(count_values);
    test_embedding(argv[1], counts);
    test_rms(argv[2], counts);
    test_residual(argv[3], counts);
    test_rope(argv[4], counts);
    test_qkv(argv[5], counts);
    test_dense_family(
        argv[6],
        "lunaflux_luna_dense_projection_bf16_reference_v1_op_4_ep_5",
        4, 7, "dense");
    test_mlp(argv[7], counts);
    test_dense_family(
        argv[8],
        "lunaflux_luna_language_model_head_bf16_reference_v1_op_6_ep_7",
        16, 8, "lm_head");
    counts.close();
    cuda_ok(cuCtxSetCurrent(nullptr), "cuCtxSetCurrent(clear)");
    cuda_ok(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    context = nullptr;
    std::cout << "outcome=bf16-family-sm120-pass families=8 live_tokens=3\n";
    return 0;
  } catch (const std::exception &error) {
    if (context != nullptr) {
      (void)cuCtxSetCurrent(nullptr);
      CUdevice device = 0;
      (void)cuDeviceGet(&device, 0);
      (void)cuDevicePrimaryCtxRelease(device);
    }
    std::cerr << "bf16-family-probe: " << error.what() << '\n';
    return 1;
  }
}
