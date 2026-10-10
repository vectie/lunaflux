// Offline ASan/LSan exercise of the same native ATen ownership implementation.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <ATen/Context.h>

extern "C" {
void *lfr_ref_open(int, int64_t);
void lfr_ref_close(void *);
const char *lfr_ref_error(void *);
int64_t lfr_ref_live(void *);
void lfr_ref_release(void *, void *);
void *lfr_ref_load(void *, const char *, int64_t, int, int);
void *lfr_ref_apply(void *, int, void *, void *, void *, int, int, int, double);
int lfr_ref_write(void *, void *, const char *);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  assert(!lfr_ref_open(0, 0));
  void *ctx = lfr_ref_open(0, 2097152);
  assert(ctx);
  void *x = lfr_ref_apply(ctx, 9, nullptr, nullptr, nullptr, 2, 4, 0, 2.0);
  void *w = lfr_ref_apply(ctx, 9, nullptr, nullptr, nullptr, 1, 4, 0, 1.0);
  assert(x && w && lfr_ref_live(ctx) == 24);
  void *n = lfr_ref_apply(ctx, 0, x, w, nullptr, 0, 0, 0, 1e-6);
  assert(n && lfr_ref_live(ctx) == 40);
  const auto original_reduction = at::globalContext().allowBF16ReductionCuBLAS();
  for (int op : {10, 11, 12}) {
    auto *linear = lfr_ref_apply(ctx, op, x, w, nullptr, 0, 0, 0, 0);
    assert(linear && lfr_ref_live(ctx) == 44);
    lfr_ref_release(ctx, linear);
    assert(at::globalContext().allowBF16ReductionCuBLAS() == original_reduction);
  }
  // Restore both reduced-precision and split-K flags under all prior modes.
  for (const auto previous : {
      at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK,
      at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK,
      at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK}) {
    at::globalContext().setAllowBF16ReductionCuBLAS(
      previous == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK,
      previous != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK);
    auto *linear = lfr_ref_apply(ctx, 12, x, w, nullptr, 0, 0, 0, 0);
    assert(linear && at::globalContext().allowBF16ReductionCuBLAS() == previous);
    lfr_ref_release(ctx, linear);
    assert(!lfr_ref_apply(ctx, 12, nullptr, w, nullptr, 0, 0, 0, 0));
    assert(at::globalContext().allowBF16ReductionCuBLAS() == previous);
    assert(lfr_ref_live(ctx) == 40);
  }
  at::globalContext().setAllowBF16ReductionCuBLAS(
    original_reduction == at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK,
    original_reduction != at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK);
  std::string path = std::string(argv[1]) + "/normalized.bf16";
  assert(lfr_ref_write(ctx, n, path.c_str()) == 0);
  assert(lfr_ref_write(ctx, n, path.c_str()) == -1); // No silent overwrite.
  assert(!lfr_ref_load(ctx, path.c_str(), 0, 4, 4)); // Short read rollback.
  assert(strstr(lfr_ref_error(ctx), "incomplete"));
  assert(lfr_ref_live(ctx) == 40);
  assert(!lfr_ref_apply(ctx, 9, nullptr, nullptr, nullptr, 65536, 65536, 0, 0));
  assert(strstr(lfr_ref_error(ctx), "budget"));
  assert(lfr_ref_live(ctx) == 40);
  auto *copy = lfr_ref_load(ctx, path.c_str(), 0, 2, 4);
  assert(copy && lfr_ref_live(ctx) == 56);
  lfr_ref_release(ctx, copy);
  lfr_ref_release(ctx, n);
  lfr_ref_release(ctx, w);
  lfr_ref_release(ctx, x);
  lfr_ref_release(ctx, x); // Idempotent registry release, no dereference.
  assert(lfr_ref_live(ctx) == 0);
  lfr_ref_close(ctx);
  // Closing the deterministic session also clears pending owners on failure.
  auto *failed = lfr_ref_open(0, 128);
  assert(lfr_ref_apply(failed, 9, nullptr, nullptr, nullptr, 2, 4, 0, 1.0));
  lfr_ref_close(failed);
  puts("reference_native_asan_probe_passed explicit_release=0 failure_owner_cleanup=checked");
}
