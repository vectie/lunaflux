#include "retained_rows_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr) do { const cudaError_t e = (expr); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); std::exit(2); \
} } while (0)

template<class T> T *upload(const std::vector<T>& data) {
  T *p; CK(cudaMalloc(&p, data.size() * sizeof(T)));
  CK(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template<class T> void read(std::vector<T>& data, T *p) {
  CK(cudaMemcpy(data.data(), p, data.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

void test(int capacity) {
  constexpr int width = 16, rows = 2, stride = 4;
  std::vector<uint16_t> input(rows * width), cache(std::max(1, capacity) * width, 0xa55a), expected = cache;
  std::vector<int32_t> counts(5), positions(rows), reset(1), upstream(3), history(2), published(5), append(3);
  auto x = upload(input), kv = upload(cache);
  auto c = upload(counts), p = upload(positions), r = upload(reset), u = upload(upstream);
  auto h = upload(history), n = upload(published), a = upload(append);
  int length = 0, poisoned = 0, steps = 0;
  auto run = [&](int live, int reset_value, int gap_row, int upstream_error, int active) {
    counts[2] = active; counts[3] = live; reset[0] = reset_value; upstream[2] = upstream_error;
    for (int row = 0; row < rows; ++row) {
      positions[row] = ((reset_value == 1 ? 0 : length) + row) * stride;
      if (row == gap_row) positions[row] += stride;
      for (int col = 0; col < width; ++col) input[row * width + col] = uint16_t(steps * 97 + row * width + col);
    }
    CK(cudaMemcpy(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c, counts.data(), 20, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(r, reset.data(), 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(u, upstream.data(), 12, cudaMemcpyHostToDevice));
    if (capacity) {
      retained3_reserve<<<1,1>>>(c,p,r,u,h,n,a);
      retained3_copy<<<rows,256>>>(a,x,kv);
      retained3_publish<<<1,1>>>(a,h,n);
    } else {
      retained0_reserve<<<1,1>>>(c,p,r,u,h,n,a);
      retained0_copy<<<rows,256>>>(a,x,kv);
      retained0_publish<<<1,1>>>(a,h,n);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    if (reset_value == 1) { length = 0; poisoned = 0; }
    const int base = length;
    const bool valid = reset_value >= 0 && reset_value <= 1 && !upstream_error && !poisoned &&
      live >= 0 && live <= rows && live <= capacity - length && !(live > 0 && active != 1) &&
      !(gap_row >= 0 && gap_row < live);
    if (valid) {
      std::copy_n(input.begin(), live * width, expected.begin() + length * width);
      length += live;
    } else poisoned = 1;
    read(cache, kv); read(history, h); read(published, n); read(append, a);
    if (cache != expected || history[0] != length || history[1] != poisoned ||
        append[0] != base || append[1] != (valid ? live : 0) || append[2] != !valid ||
        published[0] || published[1] || published[4] ||
        published[2] != (valid && length ? 1 : 0) || published[3] != (valid ? length : 0)) {
      std::fprintf(stderr, "retained rows mismatch capacity=%d step=%d\n", capacity, steps); std::exit(3);
    }
    ++steps;
  };
  run(0,1,-1,0,0); run(1,0,-1,0,1); run(0,0,-1,0,0);
  if (capacity) {
    run(2,0,-1,0,1); run(1,0,-1,0,1); run(0,0,-1,0,0);
    run(1,1,-1,0,1); run(2,0,1,0,1); run(1,0,-1,0,1);
    run(0,1,-1,0,0); run(1,0,-1,1,1); run(1,1,-1,0,1);
    run(1,0,-1,0,0); run(0,1,-1,0,0); run(-1,0,-1,0,1);
    run(0,1,-1,0,0); run(3,0,-1,0,1); run(0,2,-1,0,0);
    run(2,1,-1,0,1);
  } else run(0,1,-1,0,0);
  for (void *ptr : {(void*)x,(void*)kv,(void*)c,(void*)p,(void*)r,(void*)u,(void*)h,(void*)n,(void*)a}) CK(cudaFree(ptr));
  std::printf("retained capacity=%d steps=%d exact_words=passed publication=passed sticky_errors=passed\n", capacity, steps);
}

int main() { test(3); test(0); return 0; }
