// Offline, bounded A/B probe. Each candidate supplies its own launch geometry;
// larger compiler tiles must not accidentally inherit an older launch grid.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(expr) do { auto status = (expr); if (status != 0) { \
  std::fprintf(stderr, "CUDA %d at line %d\n", int(status), __LINE__); \
  std::exit(2); } } while (0)

struct Buffer {
  void* pointer = nullptr;
  size_t bytes;
  explicit Buffer(size_t n) : bytes(n) {
    CK(cudaMalloc(&pointer, bytes));
    CK(cudaMemset(pointer, 0, bytes));
  }
  ~Buffer() { CK(cudaFree(pointer)); }
  std::vector<__nv_bfloat16> fill(unsigned salt) {
    std::vector<__nv_bfloat16> result(bytes / 2);
    for (size_t i = 0; i < result.size(); ++i)
      result[i] = __float2bfloat16_rn(float(int((i * 17 + salt) % 31) - 15) / 32.f);
    CK(cudaMemcpy(pointer, result.data(), bytes, cudaMemcpyHostToDevice));
    return result;
  }
  std::vector<unsigned char> read() const {
    std::vector<unsigned char> result(bytes);
    CK(cudaMemcpy(result.data(), pointer, bytes, cudaMemcpyDeviceToHost));
    return result;
  }
};

struct Geometry { unsigned grid, threads, shared; };

static unsigned number(const char* text, unsigned ceiling, bool zero = false) {
  char* end = nullptr;
  unsigned long value = std::strtoul(text, &end, 10);
  if (!text[0] || *end || value > ceiling || (!zero && !value)) std::exit(1);
  return unsigned(value);
}

static Geometry geometry(char** args, int index, bool absent) {
  return { number(args[index], 1048576, absent),
           number(args[index + 1], 1024, absent),
           number(args[index + 2], 131072, true) };
}

struct Module {
  CUmodule module;
  CUfunction primary, secondary = nullptr;
  Geometry first, second;
  Module(const char* path, bool mlp, Geometry a, Geometry b) : first(a), second(b) {
    CK(cuModuleLoad(&module, path));
    CK(cuModuleGetFunction(&primary, module, mlp
      ? "lunaflux_luna_gated_mlp_bf16_release_v1"
      : "lunaflux_luna_dense_projection_bf16_release_v1"));
    if (mlp) CK(cuModuleGetFunction(&secondary, module,
      "lunaflux_luna_gated_mlp_bf16_release_v1_down"));
    describe(path, "primary", primary, first);
    if (mlp) describe(path, "down", secondary, second);
  }
  ~Module() { CK(cuModuleUnload(module)); }
  static void describe(const char* path, const char* part, CUfunction f, Geometry g) {
    int registers = 0, static_shared = 0;
    CK(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, f));
    CK(cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));
    if (g.shared + unsigned(static_shared) > 49152)
      CK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           int(g.shared)));
    std::printf("module=%s part=%s grid=%u block=%u registers=%d static_shared=%d dynamic_shared=%u\n",
      path, part, g.grid, g.threads, registers, static_shared, g.shared);
  }
  static void launch_one(CUfunction f, Geometry g, void** args) {
    CK(cuLaunchKernel(f, g.grid, 1, 1, g.threads, 1, 1, g.shared, nullptr, args, nullptr));
  }
  void launch(void** args, int part) {
    if (part != 2) launch_one(primary, first, args);
    if (secondary && part != 1) launch_one(secondary, second, args);
  }
  float time(void** args, int part) {
    for (int i = 0; i < 3; ++i) launch(args, part);
    CK(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i) launch(args, part);
    CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
    float millis = 0; CK(cudaEventElapsedTime(&millis, begin, end));
    CK(cudaEventDestroy(begin)); CK(cudaEventDestroy(end));
    return millis * 50;
  }
};

static float at(const std::vector<unsigned char>& bytes, size_t index) {
  __nv_bfloat16 value;
  std::memcpy(&value, bytes.data() + index * 2, 2);
  return __bfloat162float(value);
}

int main(int argc, char** argv) {
  if (argc != 18 && argc != 19) {
    std::fprintf(stderr, "KIND OLD NEW TOKENS ROWS OLD_PRIMARY(grid block shared) OLD_DOWN(grid block shared) NEW_PRIMARY(grid block shared) NEW_DOWN(grid block shared) [--check]\n");
    return 1;
  }
  const bool mlp = std::strcmp(argv[1], "mlp") == 0;
  if (!mlp && std::strcmp(argv[1], "output") != 0) return 1;
  const unsigned tokens = number(argv[4], 2048), rows = number(argv[5], 32);
  if (tokens < rows || (argc == 19 && std::strcmp(argv[18], "--check") != 0)) return 1;
  const unsigned hidden = 1024, intermediate = 3072, input_width = mlp ? hidden : 2048;
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  Buffer counts(20), input(size_t(2048) * input_width * 2),
    gate(size_t(mlp ? intermediate : hidden) * input_width * 2),
    up(size_t(intermediate) * hidden * 2), down(size_t(intermediate) * hidden * 2),
    output(size_t(2048) * hidden * 2), workspace(size_t(2048) * intermediate * 2);
  int live[5] = {int(rows), 0, int(rows), int(tokens), int(rows)};
  CK(cudaMemcpy(counts.pointer, live, sizeof(live), cudaMemcpyHostToDevice));
  auto x = input.fill(3), g = gate.fill(7), u = up.fill(11), d = down.fill(13);
  Module old(argv[2], mlp, geometry(argv, 6, false), geometry(argv, 9, !mlp));
  Module now(argv[3], mlp, geometry(argv, 12, false), geometry(argv, 15, !mlp));
  void* mlp_args[] = {&counts.pointer, &input.pointer, &gate.pointer, &up.pointer,
                     &down.pointer, &output.pointer, &workspace.pointer};
  void* output_args[] = {&counts.pointer, &input.pointer, &gate.pointer, &output.pointer};
  void** args = mlp ? mlp_args : output_args;
  old.launch(args, 0); CK(cudaDeviceSynchronize());
  auto expected = output.read(), expected_workspace = workspace.read();
  CK(cudaMemset(output.pointer, 0, output.bytes));
  CK(cudaMemset(workspace.pointer, 0, workspace.bytes));
  now.launch(args, 0); CK(cudaDeviceSynchronize());
  auto actual = output.read(), actual_workspace = workspace.read();
  if (expected != actual || (mlp && expected_workspace != actual_workspace)) {
    std::fprintf(stderr, "ordered complete-chain output mismatch\n"); return 3;
  }
  double max_error = 0;
  for (unsigned row : {0u, tokens / 2, tokens - 1}) {
    if (mlp) for (unsigned col : {0u, 15u, 16u, 1023u, 2048u, intermediate - 1}) {
      float a = 0, b = 0;
      for (unsigned k = 0; k < hidden; ++k) {
        a += __bfloat162float(x[size_t(row) * hidden + k]) * __bfloat162float(g[size_t(col) * hidden + k]);
        b += __bfloat162float(x[size_t(row) * hidden + k]) * __bfloat162float(u[size_t(col) * hidden + k]);
      }
      float reference = __bfloat162float(__float2bfloat16_rn(a / (1 + std::exp(-a)) * b));
      double error = std::abs(double(at(actual_workspace, size_t(row) * intermediate + col) - reference));
      if (!std::isfinite(error) || error > 0.02) return 4;
      max_error = std::max(max_error, error);
    }
    for (unsigned col : {0u, 15u, 16u, 511u, hidden - 1}) {
      float sum = 0;
      const unsigned inner = mlp ? intermediate : input_width;
      for (unsigned k = 0; k < inner; ++k) {
        const float activation = mlp ? at(actual_workspace, size_t(row) * inner + k)
                                    : __bfloat162float(x[size_t(row) * inner + k]);
        const float weight = __bfloat162float((mlp ? d : g)[size_t(col) * inner + k]);
        sum += activation * weight;
      }
      float reference = __bfloat162float(__float2bfloat16_rn(sum));
      double error = std::abs(double(at(actual, size_t(row) * hidden + col) - reference));
      if (!std::isfinite(error) || error > 0.02) return 5;
      max_error = std::max(max_error, error);
    }
  }
  std::printf("correctness=passed bitwise_chain=true scalar_maxabs=%.9g\n", max_error);
  if (argc == 19) return 0;
  for (int part : mlp ? std::vector<int>{1, 2, 0} : std::vector<int>{0})
    for (int trial = 0; trial < 5; ++trial) {
      float before, after;
      if (trial % 2) { after = now.time(args, part); before = old.time(args, part); }
      else { before = old.time(args, part); after = now.time(args, part); }
      std::printf("tokens=%u rows=%u part=%d trial=%d old_us=%.6f new_us=%.6f\n",
        tokens, rows, part, trial, before, after);
    }
}
