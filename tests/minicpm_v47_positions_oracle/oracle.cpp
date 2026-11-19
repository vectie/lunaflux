// Offline reference only: Transformers 5.18.0 MiniCPMV4_7Model's
// get_vision_position_ids uses these ATen operations. No model weights needed.
#include <ATen/ATen.h>
#include <torch/version.h>
#include <fstream>
#include <stdexcept>

static void emit(std::ostream& out, int count, int extent) {
  auto values = at::linspace(0, extent - 1, count,
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
      .round().to(at::kLong).contiguous();
  const auto* coordinates = values.const_data_ptr<int64_t>();
  out << "{\"count\":" << count << ",\"extent\":" << extent
      << ",\"coordinates\":[";
  for (int i = 0; i < count; ++i) {
    if (i) out << ',';
    out << coordinates[i];
  }
  out << "]}\n";
}

int main(int argc, char** argv) {
  if (argc != 2) throw std::runtime_error("oracle NEW_OUTPUT_JSONL");
  std::ofstream out(argv[1]);
  if (!out) throw std::runtime_error("cannot open output");
  out << "{\"oracle\":\"ATen FP32 linspace/round/long CPU\",\"torch_version\":\""
      << TORCH_VERSION << "\"}\n";
  for (int count = 1; count <= 64; ++count)
    for (int extent = 1; extent <= 128; ++extent) emit(out, count, extent);
  for (int count : {127, 128})
    for (int extent : {511, 512}) emit(out, count, extent);
  if (!out) throw std::runtime_error("cannot write oracle output");
}
