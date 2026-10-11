// Independent CPU and real-device arithmetic smoke; no checkpoint rewriting.
#include <ATen/ATen.h>
#include <cassert>
#include <cstdio>

namespace lfr {
at::Tensor nvfp4_decode(const at::Tensor &, const at::Tensor &, const at::Tensor &);
at::Tensor nvfp4_experts(const at::Tensor &, const at::Tensor &, const at::Tensor &,
 const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
 const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, double);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  const at::Device device(argv[1]);
  auto f = at::TensorOptions().device(device).dtype(at::kFloat);
  auto packed = at::arange(8, at::TensorOptions().dtype(at::kByte)).reshape({1, 8});
  packed = (packed * 16 + packed).to(device);
  auto scales = at::ones({1, 1}, f).to(at::kFloat8_e4m3fn);
  auto decoded = lfr::nvfp4_decode(packed, scales, at::full({}, 2., f)).to(at::kCPU, at::kFloat);
  auto expected = at::tensor({0.f,0.f,1.f,1.f,2.f,2.f,3.f,3.f,4.f,4.f,6.f,6.f,8.f,8.f,12.f,12.f}).reshape({1,16});
  assert(at::equal(decoded, expected));
  auto negative = lfr::nvfp4_decode(at::bitwise_or(packed, 136), scales, at::full({}, 2., f));
  assert(at::equal(negative.to(at::kCPU, at::kFloat), -expected));
  auto byte = at::TensorOptions().device(device).dtype(at::kByte);
  auto x = at::full({2,16}, .25, f).to(at::kBFloat16);
  auto w13 = at::full({2,32,8}, 34, byte); // Every nibble is +1.
  auto s13 = at::ones({2,32,1}, f).to(at::kFloat8_e4m3fn);
  auto g13 = at::tensor({.5f,2.f,1.f,.25f}).reshape({2,2}).to(device);
  auto w2 = at::full({2,16,8}, 34, byte);
  auto s2 = at::ones({2,16,1}, f).to(at::kFloat8_e4m3fn);
  auto g2 = at::tensor({.25f,.5f}).to(device);
  auto ids = at::tensor({1L,0L,0L,1L}).reshape({2,2}).to(device);
  auto scores = at::tensor({.25f,.75f,.75f,.25f}).reshape({2,2}).to(device);
  auto actual = lfr::nvfp4_experts(x,w13,s13,g13,w2,s2,g2,scores,ids,c10::nullopt,0.);
  auto first = ((at::silu(at::full({2,16},2.,f).to(at::kBFloat16)) * 8).sum(1,true) * .25).to(at::kBFloat16);
  auto second = ((at::silu(at::full({2,16},4.,f).to(at::kBFloat16)) * 1).sum(1,true) * .5).to(at::kBFloat16);
  auto target = ((first.to(at::kFloat)*.75).to(at::kBFloat16).to(at::kFloat) +
                 (second.to(at::kFloat)*.25).to(at::kBFloat16).to(at::kFloat)).to(at::kBFloat16);
  assert(at::equal(actual, target.expand({2,16})));
  auto collapsed = g13.clone(); collapsed.select(1,1).copy_(collapsed.select(1,0));
  auto wrong = lfr::nvfp4_experts(x,w13,s13,collapsed,w2,s2,g2,scores,ids,c10::nullopt,0.);
  assert(!at::equal(actual, wrong));
  puts("nvfp4_reference_probe_passed separate_gate_up_scale=true original_nibbles=true bf16_boundaries=true");
}
