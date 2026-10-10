// Independent diagnostic lowering of the upstream H3 encoder arithmetic.
// All layer planning and resource scopes remain in the MoonBit driver.
// No production package links this file or ATen.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <ATen/core/grad_mode.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>

struct Context {
  at::Device device;
  int64_t budget, live = 0;
  std::string error;
  std::unordered_map<at::Tensor *, int64_t> tensors;
  Context(int gpu, int64_t limit) : device(gpu ? at::kCUDA : at::kCPU), budget(limit) {}
  ~Context() { for (auto &item : tensors) delete item.first; }
  void reserve(int64_t bytes) const {
    if (bytes < 0 || bytes > budget - live) throw std::runtime_error("reference tensor budget exceeded");
  }
  at::Tensor *own(at::Tensor tensor) {
    int64_t bytes = (int64_t)tensor.nbytes();
    reserve(bytes);
    auto *result = new at::Tensor(std::move(tensor));
    try { tensors.emplace(result, bytes); } catch (...) { delete result; throw; }
    live += bytes;
    return result;
  }
  const at::Tensor &get(void *p) const {
    auto *tensor = static_cast<at::Tensor *>(p);
    if (!tensor || !tensors.count(tensor)) throw std::runtime_error("foreign/closed tensor");
    return *tensor;
  }
};

extern "C" void *lfr_ref_open(int gpu, int64_t budget) {
  try {
    if ((gpu != 0 && gpu != 1) || budget <= 0 || budget > 8589934592LL) return nullptr;
    at::set_num_threads(2);
    return new Context(gpu, budget);
  } catch (...) { return nullptr; }
}
extern "C" void lfr_ref_close(void *p) { delete static_cast<Context *>(p); }
extern "C" const char *lfr_ref_error(void *p) { return static_cast<Context *>(p)->error.c_str(); }
extern "C" int64_t lfr_ref_live(void *p) { return static_cast<Context *>(p)->live; }
extern "C" void lfr_ref_release(void *p, void *tensor) {
  auto &ctx = *static_cast<Context *>(p);
  auto it = ctx.tensors.find(static_cast<at::Tensor *>(tensor));
  if (it != ctx.tensors.end()) {
    ctx.live -= it->second;
    delete it->first;
    ctx.tensors.erase(it);
  }
}

struct File {
  int fd;
  File(const char *path, int flags) : fd(open(path, flags, 0600)) {
    if (fd < 0) throw std::runtime_error(std::string(path) + ": " + strerror(errno));
  }
  ~File() { close(fd); }
};

extern "C" void *lfr_ref_load(void *p, const char *path, int64_t offset, int rows, int cols) {
  auto &ctx = *static_cast<Context *>(p);
  ctx.error.clear();
  try {
    at::NoGradGuard no_grad;
    if (rows <= 0 || cols <= 0 || offset < 0) throw std::runtime_error("invalid BF16 region");
    int64_t bytes = (int64_t)rows * cols * 2;
    ctx.reserve(bytes);
    File file(path, O_RDONLY);
    auto cpu = at::empty({rows, cols}, at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU));
    auto *data = static_cast<uint8_t *>(cpu.data_ptr());
    for (int64_t done = 0; done < bytes;) {
      ssize_t n = pread(file.fd, data + done, (size_t)std::min<int64_t>(bytes - done, 8388608), offset + done);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("incomplete original weight region");
      done += n;
    }
    return ctx.own(cpu.to(ctx.device, at::kBFloat16, false, true));
  } catch (const std::exception &e) { ctx.error = e.what(); return nullptr; }
}

extern "C" void *lfr_ref_apply(void *p, int op, void *xp, void *yp, void *zp, int a, int b, int c, double d) {
  auto &ctx = *static_cast<Context *>(p);
  ctx.error.clear();
  try {
    at::NoGradGuard no_grad;
    at::Tensor out;
    switch (op) {
      case 0: {
        auto &x = ctx.get(xp); auto &weight = ctx.get(yp);
        auto xf = x.reshape({-1, weight.numel()}).to(at::kFloat);
        auto normalized = xf * at::rsqrt((xf * xf).mean(-1, true) + d);
        out = (normalized.to(at::kBFloat16) * weight.reshape({1, -1})).reshape(x.sizes());
        break;
      }
      case 1: out = at::linear(ctx.get(xp), ctx.get(yp)); break;
      case 2: case 3: {
        if (a <= 0 || b <= 0 || b % 2) throw std::runtime_error("invalid rotary extent");
        auto options = at::TensorOptions().dtype(at::kFloat).device(ctx.device);
        auto indices = at::arange(0, b, 2, options) / b;
        auto inv = 1.0 / at::pow(d, indices);
        auto positions = at::arange(a, options);
        auto frequencies = at::matmul(inv.reshape({-1, 1}), positions.reshape({1, -1})).transpose(0, 1);
        auto angle = at::cat({frequencies, frequencies}, -1);
        out = (op == 2 ? at::cos(angle) : at::sin(angle)).to(at::kBFloat16);
        break;
      }
      case 4: {
        auto &x = ctx.get(xp);
        auto rows = x.size(0);
        auto shaped = x.reshape({rows, a, b});
        auto rotated = at::cat({-shaped.slice(-1, b / 2, b), shaped.slice(-1, 0, b / 2)}, -1);
        // Keep BF16 rounding at each product and at the sum, as in upstream.
        out = (shaped * ctx.get(yp).unsqueeze(1) + rotated * ctx.get(zp).unsqueeze(1)).reshape(x.sizes());
        break;
      }
      case 5: {
        auto &q = ctx.get(xp); auto rows = q.size(0);
        if (a <= 0 || b <= 0 || c <= 0 || a % b) throw std::runtime_error("invalid attention heads");
        auto qs = q.reshape({1, rows, a, c}).transpose(1, 2);
        auto ks = ctx.get(yp).reshape({1, rows, b, c}).transpose(1, 2).repeat_interleave(a / b, 1);
        auto vs = ctx.get(zp).reshape({1, rows, b, c}).transpose(1, 2).repeat_interleave(a / b, 1);
        out = at::scaled_dot_product_attention(qs, ks, vs, {}, 0.0, true).transpose(1, 2).reshape({rows, a * c}).contiguous();
        break;
      }
      case 6: out = at::silu(ctx.get(xp)) * ctx.get(yp); break;
      case 7: out = ctx.get(xp) + ctx.get(yp); break;
      case 8: out = at::cat({ctx.get(xp), ctx.get(yp)}, 0); break;
      case 9: {
        if (a <= 0 || b <= 0) throw std::runtime_error("invalid constant shape");
        ctx.reserve((int64_t)a * b * 2);
        out = at::full({a, b}, d, at::TensorOptions().dtype(at::kBFloat16).device(ctx.device));
        break;
      }
      default: throw std::runtime_error("unknown reference operator");
    }
    return ctx.own(std::move(out));
  } catch (const std::exception &e) { ctx.error = e.what(); return nullptr; }
}

extern "C" int lfr_ref_write(void *p, void *tensor, const char *path) {
  auto &ctx = *static_cast<Context *>(p);
  ctx.error.clear();
  try {
    auto cpu = ctx.get(tensor).to(at::kCPU).contiguous();
    File file(path, O_WRONLY | O_CREAT | O_EXCL);
    auto *data = static_cast<const uint8_t *>(cpu.const_data_ptr());
    int64_t bytes = (int64_t)cpu.nbytes();
    for (int64_t done = 0; done < bytes;) {
      ssize_t n = write(file.fd, data + done, (size_t)(bytes - done));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("incomplete BF16 output write");
      done += n;
    }
    return 0;
  } catch (const std::exception &e) { ctx.error = e.what(); return -1; }
}
