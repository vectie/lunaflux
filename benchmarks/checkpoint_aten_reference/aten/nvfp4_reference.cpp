// Test-only original-layout NVFP4 arithmetic. Never linked into LunaFlux.
// Keep gate/up global scales distinct and decode one selected expert at a time.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/silu.h>
#include <torch/library.h>
#include <algorithm>
#include <vector>

namespace lfr {
at::Tensor nvfp4_decode(const at::Tensor &packed, const at::Tensor &scales,
                       const at::Tensor &global) {
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "original row-major packed NVFP4 weight required");
  const auto rows = packed.size(0), columns = packed.size(1) * 2;
  TORCH_CHECK(columns % 16 == 0 && scales.dim() == 2 &&
              scales.size(0) == rows && scales.size(1) == columns / 16 &&
              scales.scalar_type() == at::kFloat8_e4m3fn && global.numel() == 1,
              "NVFP4 requires block-16 E4M3 and one projection global scale");
  // Independent ATen indexing, not production decode tables/layout helpers.
  auto table = at::tensor({0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                          -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f},
                         at::TensorOptions().dtype(at::kFloat)).to(packed.device());
  auto low = at::bitwise_and(packed, 15).to(at::kLong);
  auto high = at::bitwise_right_shift(packed, 4).to(at::kLong);
  auto values = at::stack({table.index_select(0, low.flatten()).reshape(packed.sizes()),
                           table.index_select(0, high.flatten()).reshape(packed.sizes())}, -1)
                    .reshape({rows, columns / 16, 16});
  // F32 block multiply, then separate F32 global multiply, then BF16 operand.
  return ((values * scales.to(at::kFloat).unsqueeze(-1)) * global.to(at::kFloat))
      .reshape({rows, columns}).to(at::kBFloat16);
}

at::Tensor nvfp4_experts(const at::Tensor &x, const at::Tensor &w13,
                        const at::Tensor &s13, const at::Tensor &g13,
                        const at::Tensor &w2, const at::Tensor &s2,
                        const at::Tensor &g2, const at::Tensor &weights,
                        const at::Tensor &ids, const c10::optional<at::Tensor> &map,
                        double clamp) {
  at::NoGradGuard no_grad;
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 &&
              x.size(0) <= 256 && w13.dim() == 3 && w2.dim() == 3 &&
              weights.dim() == 2 && ids.sizes() == weights.sizes() &&
              ids.size(0) == x.size(0) && ids.size(1) <= 32,
              "bounded BF16 selected-expert reference required");
  const int64_t experts = w13.size(0), intermediate = w13.size(1) / 2;
  TORCH_CHECK(w13.size(1) == intermediate * 2 &&
              w13.size(2) * 2 == x.size(1) && w2.size(0) == experts &&
              w2.size(1) == x.size(1) && w2.size(2) * 2 == intermediate &&
              s13.sizes() == at::IntArrayRef({experts, intermediate * 2, x.size(1) / 16}) &&
              s2.sizes() == at::IntArrayRef({experts, x.size(1), intermediate / 16}) &&
              g13.sizes() == at::IntArrayRef({experts, 2}) && g2.numel() == experts &&
              x.size(1) % 16 == 0 && intermediate % 16 == 0,
              "original gate/up/down weight geometry required");
  // Conservative transient bound: one decoded projection and ATen's F32
  // decoding/index temporaries, plus the small selected-token outputs.
  TORCH_CHECK(std::max(intermediate * x.size(1), x.size(1) * intermediate) * 48 +
              x.numel() * ids.size(1) * 8 <= 2147483648LL,
              "reference expert transient allocation exceeds 2 GiB");
  auto selected = ids.to(at::kCPU, at::kLong).contiguous();
  auto local_map = map.has_value() ? map->to(at::kCPU, at::kLong).contiguous() : at::Tensor();
  auto *p = selected.const_data_ptr<int64_t>();
  auto output = at::zeros(x.sizes(), x.options().dtype(at::kFloat));
  std::vector<int64_t> active(p, p + selected.numel());
  std::sort(active.begin(), active.end());
  active.erase(std::unique(active.begin(), active.end()), active.end());
  for (const auto expert : active) {
    TORCH_CHECK(expert >= 0, "negative routed expert");
    const auto local = local_map.defined()
        ? (expert < local_map.numel() ? local_map.const_data_ptr<int64_t>()[expert] : -1)
        : expert;
    if (local < 0) continue;
    TORCH_CHECK(local < experts, "selected expert outside original weight bank");
    std::vector<int64_t> token_rows, slots;
    for (int64_t row = 0; row < selected.size(0); ++row)
      for (int64_t slot = 0; slot < selected.size(1); ++slot)
        if (p[row * selected.size(1) + slot] == expert) {
          token_rows.push_back(row); slots.push_back(slot);
        }
    auto rows = at::tensor(token_rows, at::TensorOptions().dtype(at::kLong)).to(x.device());
    auto slot_indices = at::tensor(slots, at::TensorOptions().dtype(at::kLong)).to(x.device());
    auto input = x.index_select(0, rows);
    auto gate = at::linear(input, nvfp4_decode(w13[local].narrow(0, 0, intermediate),
        s13[local].narrow(0, 0, intermediate), g13[local][0]));
    auto up = at::linear(input, nvfp4_decode(w13[local].narrow(0, intermediate, intermediate),
        s13[local].narrow(0, intermediate, intermediate), g13[local][1]));
    if (clamp > 0) { gate = gate.clamp_max(clamp); up = up.clamp(-clamp, clamp); }
    // Preserve BF16 SiLU, product, down output and weighted-output boundaries.
    auto product = at::silu(gate) * up;
    auto down = at::linear(product, nvfp4_decode(w2[local], s2[local], g2[local]));
    auto score = weights.index({rows, slot_indices}).to(at::kFloat).unsqueeze(-1);
    auto weighted = (down.to(at::kFloat) * score).to(at::kBFloat16).to(at::kFloat);
    output.index_add_(0, rows, weighted);
  }
  return output.to(at::kBFloat16);
}
} // namespace lfr

TORCH_LIBRARY(lunaflux_reference, m) {
  m.def("nvfp4_decode(Tensor packed, Tensor scales, Tensor global_scale) -> Tensor");
  m.def("nvfp4_experts(Tensor x, Tensor w13, Tensor s13, Tensor g13, Tensor w2, Tensor s2, Tensor g2, Tensor weights, Tensor ids, Tensor? map, float clamp) -> Tensor");
}
TORCH_LIBRARY_IMPL(lunaflux_reference, CompositeExplicitAutograd, m) {
  m.impl("nvfp4_decode", TORCH_FN(lfr::nvfp4_decode));
  m.impl("nvfp4_experts", TORCH_FN(lfr::nvfp4_experts));
}
