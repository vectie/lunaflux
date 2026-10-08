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
#include <string>
#include <memory>
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
  Module(const char* path, bool mlp, Geometry a, Geometry b, unsigned bounded_rows = 0) : first(a), second(b) {
    CK(cuModuleLoad(&module, path));
    std::string symbol = mlp
      ? "lunaflux_luna_gated_mlp_bf16_release_v1"
      : "lunaflux_luna_dense_projection_bf16_release_v1";
    if (bounded_rows) symbol += "_rows" + std::to_string(bounded_rows);
    CK(cuModuleGetFunction(&primary, module, symbol.c_str()));
    if (mlp) CK(cuModuleGetFunction(&secondary, module, (symbol + "_down").c_str()));
    std::printf("selected_symbol=%s\n", symbol.c_str());
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
  float time(void** args, int part, const std::vector<void*>& weights,
             const std::vector<void*>& gate_weights,
             const std::vector<void*>& up_weights) {
    for (int i = 0; i < 3; ++i) launch(args, part);
    CK(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    CK(cudaEventCreate(&begin)); CK(cudaEventCreate(&end));
    CK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i) {
      if (weights.empty()) launch(args, part);
      else {
        void* current[7];
        for (int j = 0; j < 7; ++j) current[j] = args[j];
        current[4] = const_cast<void**>(&weights[size_t(i) % weights.size()]);
        if (!gate_weights.empty()) {
          current[2] = const_cast<void**>(&gate_weights[size_t(i) % gate_weights.size()]);
          current[3] = const_cast<void**>(&up_weights[size_t(i) % up_weights.size()]);
        }
        launch(current, part);
      }
    }
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
  if (argc < 18 || argc > 23) {
    std::fprintf(stderr, "KIND OLD NEW TOKENS ROWS OLD_PRIMARY(grid block shared) OLD_DOWN(grid block shared) NEW_PRIMARY(grid block shared) NEW_DOWN(grid block shared) [--check] [--bounded-symbols] [--down-only-change|--gate-only-change] [--streaming-weights|--streaming-all-weights]\n");
    return 1;
  }
  const bool mlp = std::strcmp(argv[1], "mlp") == 0;
  if (!mlp && std::strcmp(argv[1], "output") != 0) return 1;
  // Offline chunk experiment only. The supplied AOT modules must carry this
  // capacity too; raising a probe limit cannot extend a serving bundle.
  const unsigned tokens = number(argv[4], 8192), rows = number(argv[5], 32);
  bool check_only = false, bounded_symbols = false, down_only_change = false, gate_only_change = false, streaming_weights = false, streaming_all = false;
  for (int i = 18; i < argc; ++i) {
    if (std::strcmp(argv[i], "--check") == 0 && !check_only) check_only = true;
    else if (std::strcmp(argv[i], "--bounded-symbols") == 0 && !bounded_symbols) bounded_symbols = true;
    else if (std::strcmp(argv[i], "--down-only-change") == 0 && !down_only_change && mlp) down_only_change = true;
    else if (std::strcmp(argv[i], "--gate-only-change") == 0 && !gate_only_change && mlp) gate_only_change = true;
    else if (std::strcmp(argv[i], "--streaming-weights") == 0 && !streaming_weights && mlp) streaming_weights = true;
    else if (std::strcmp(argv[i], "--streaming-all-weights") == 0 && !streaming_weights && mlp) { streaming_weights = true; streaming_all = true; }
    else return 1;
  }
  if (tokens < rows || (bounded_symbols && tokens > 16)) return 1;
  if (gate_only_change && down_only_change) return 1;
  const unsigned hidden = 1024, intermediate = 3072, input_width = mlp ? hidden : 2048;
  CK(cudaSetDevice(0)); CK(cudaFree(nullptr));
  Buffer counts(20), input(size_t(tokens) * input_width * 2),
    gate(size_t(mlp ? intermediate : hidden) * input_width * 2),
    up(size_t(intermediate) * hidden * 2), down(size_t(intermediate) * hidden * 2),
    output(size_t(tokens) * hidden * 2), workspace(size_t(tokens) * intermediate * 2);
  int live[5] = {int(rows), 0, int(rows), int(tokens), int(rows)};
  CK(cudaMemcpy(counts.pointer, live, sizeof(live), cudaMemcpyHostToDevice));
  auto x = input.fill(3), g = gate.fill(7), u = up.fill(11), d = down.fill(13);
  const unsigned bounded_rows = bounded_symbols ? (tokens <= 8 ? 8 : 16) : 0;
  Module old(argv[2], mlp, geometry(argv, 6, false), geometry(argv, 9, !mlp), bounded_rows);
  Module now(argv[3], mlp, geometry(argv, 12, false), geometry(argv, 15, !mlp), bounded_rows);
  // Controlled diagnostic ablation: the production producer remains identical.
  // Both modules stay live through all trials; no artifact is modified.
  if (down_only_change) {
    now.primary = old.primary;
    now.first = old.first;
    std::printf("ablation=down-only-change producer=old-module\n");
  }
  if (gate_only_change) {
    now.secondary = old.secondary;
    now.second = old.second;
    std::printf("ablation=gate-only-change down=old-module\n");
  }
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
  if (check_only) return 0;
  std::vector<std::unique_ptr<Buffer>> weight_storage;
  std::vector<void*> weights;
  std::vector<void*> gate_weights, up_weights;
  if (streaming_weights) {
    // Match a decoder's distinct layer-weight working set without timing
    // uploads or cache-flush kernels. Data are identical for A/B correctness.
    for (int layer = 0; layer < 28; ++layer) {
      weight_storage.emplace_back(new Buffer(down.bytes));
      weight_storage.back()->fill(13);
      weights.push_back(weight_storage.back()->pointer);
      if (streaming_all) {
        weight_storage.emplace_back(new Buffer(gate.bytes));
        weight_storage.back()->fill(7);
        gate_weights.push_back(weight_storage.back()->pointer);
        weight_storage.emplace_back(new Buffer(up.bytes));
        weight_storage.back()->fill(11);
        up_weights.push_back(weight_storage.back()->pointer);
      }
    }
    std::printf("weight_working_set=%s bytes=%zu\n", streaming_all ? "28-distinct-gate-up-down-matrices" : "28-distinct-down-matrices", 28 * (down.bytes + (streaming_all ? gate.bytes + up.bytes : 0)));
  }
  for (int part : mlp ? std::vector<int>{1, 2, 0} : std::vector<int>{0})
    for (int trial = 0; trial < 5; ++trial) {
      float before, after;
      if (trial % 2) { after = now.time(args, part, weights, gate_weights, up_weights); before = old.time(args, part, weights, gate_weights, up_weights); }
      else { before = old.time(args, part, weights, gate_weights, up_weights); after = now.time(args, part, weights, gate_weights, up_weights); }
      std::printf("tokens=%u rows=%u part=%d trial=%d old_us=%.6f new_us=%.6f\n",
        tokens, rows, part, trial, before, after);
    }
}
