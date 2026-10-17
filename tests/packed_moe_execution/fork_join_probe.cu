// Diagnostic AOT fixture only. The real MoonBit owner, not a substitute CUDA
// dispatcher, creates and replays the fork/join graph around these kernels.
extern "C" __global__ void fork_branch(const unsigned *input, unsigned *out) {
  unsigned i=threadIdx.x;
  out[i]=input[i]+1;
}
extern "C" __global__ void fork_join(const unsigned *a,const unsigned *b,unsigned *out) {
  unsigned i=threadIdx.x;
  out[i]=a[i]+b[i];
}
