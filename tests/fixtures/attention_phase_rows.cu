// Diagnostic fixture for the real prepared phase-owner consumer. Each token
// must observe its KV write and exactly one attention writer before output.
extern "C" __global__ void phase_ingress(const int* counts, int* state) {
  for (int token = threadIdx.x; token < counts[3]; token += blockDim.x) {
    state[token] = 1000 + token;
    state[32 + token] = 0;
  }
}

extern "C" __global__ void phase_attention(const int* counts, int* state) {
  for (int token = threadIdx.x; token < counts[3]; token += blockDim.x) {
    state[token] += 100;
    state[32 + token] += 1;
  }
}

extern "C" __global__ void phase_decode(const int* counts, int* state) {
  const int start = counts[3] - counts[1];
  for (int row = threadIdx.x; row < counts[1]; row += blockDim.x) {
    const int token = start + row;
    state[token] += 100;
    state[32 + token] += 1;
  }
}

extern "C" __global__ void phase_output(const int* counts, int* state) {
  for (int token = threadIdx.x; token < counts[3]; token += blockDim.x) {
    state[64 + token] = state[token] == 1100 + token &&
                       state[32 + token] == 1 ? 1 : -1;
  }
}
