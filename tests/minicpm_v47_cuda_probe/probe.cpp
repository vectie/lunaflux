// Deployment diagnostic for the actual prepared CUDA modules. No checkpoint
// payload, fake driver, CUDA runtime JIT, Python or model-quality claim.
#include <cuda.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void checked(CUresult result) {
  if (result == CUDA_SUCCESS) return;
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  throw std::runtime_error(name ? name : "CUDA driver error");
}

struct Owned {
  CUdevice device = 0;
  bool retained = false;
  std::vector<CUmodule> modules;
  std::vector<CUdeviceptr> allocations;
  ~Owned() {
    if (!retained) return;
    cuCtxSynchronize();
    for (auto p : allocations) cuMemFree(p);
    for (auto m : modules) cuModuleUnload(m);
    cuDevicePrimaryCtxRelease(device);
  }
  CUdeviceptr upload(const void *source, size_t bytes) {
    CUdeviceptr p = 0;
    checked(cuMemAlloc(&p, bytes));
    allocations.push_back(p);
    checked(cuMemcpyHtoD(p, source, bytes));
    return p;
  }
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: minicpm-cuda-probe CUBIN_ROOT\n");
    return 2;
  }
  try {
    checked(cuInit(0));
    Owned owned;
    checked(cuDeviceGet(&owned.device, 0));
    int major = 0, minor = 0;
    checked(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, owned.device));
    checked(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, owned.device));
    if (major != 12 || minor != 1) throw std::runtime_error("expected sm121");
    CUcontext context = nullptr;
    checked(cuDevicePrimaryCtxRetain(&context, owned.device));
    owned.retained = true;
    checked(cuCtxSetCurrent(context));
    const std::array<const char *, 9> names = {
      "decoder_envelope", "recurrent_branch", "moe", "attention_common",
      "attention_prefill", "attention_decode", "text_io", "vision_encoder", "image_scatter"
    };
    for (auto name : names) {
      CUmodule module = nullptr;
      checked(cuModuleLoad(&module, (std::string(argv[1]) + "/" + name + ".cubin").c_str()));
      owned.modules.push_back(module);
    }
    CUfunction scatter = nullptr;
    checked(cuModuleGetFunction(&scatter, owned.modules.back(), "lunaflux_minicpm_v47_image_scatter"));
    const int32_t counts[4] = {1, 0, 0, 3};
    const int32_t positions[3] = {5, 6, 7};
    const int32_t feature_count = 2;
    std::vector<int32_t> map(4096, -1);
    map[5] = 1;
    map[7] = 0;
    std::vector<uint16_t> features(2 * 2048), output(3 * 2048, 0xdead);
    for (size_t i = 0; i < features.size(); ++i) features[i] = uint16_t(0x3f00 + (i / 2048) * 32 + i % 32);
    auto a = owned.upload(counts, sizeof(counts));
    auto b = owned.upload(positions, sizeof(positions));
    auto c = owned.upload(map.data(), map.size() * sizeof(int32_t));
    auto d = owned.upload(&feature_count, sizeof(feature_count));
    auto e = owned.upload(features.data(), features.size() * sizeof(uint16_t));
    auto f = owned.upload(output.data(), output.size() * sizeof(uint16_t));
    void *arguments[] = {&a, &b, &c, &d, &e, &f};
    for (int i = 0; i < 1000; ++i) {
      checked(cuLaunchKernel(scatter, 3, 1, 1, 256, 1, 1, 0, nullptr, arguments, nullptr));
    }
    checked(cuCtxSynchronize());
    checked(cuMemcpyDtoH(output.data(), f, output.size() * sizeof(uint16_t)));
    for (size_t column = 0; column < 2048; ++column) {
      if (output[column] != features[2048 + column] ||
          output[2048 + column] != 0xdead ||
          output[4096 + column] != features[column]) {
        throw std::runtime_error("image scatter word mismatch");
      }
    }
    std::puts("{\"format\":\"lunaflux.minicpm-v47-cuda-module-probe.v1\",\"architecture\":\"sm_121\",\"modules_loaded\":9,\"scatter_launches\":1000,\"verified_words\":6144,\"gpu_execution\":true,\"complete_model_execution\":false,\"numerical_qualified\":false}");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
