extern "C" __global__ void lunaflux_physical_add_one(unsigned int *values) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 4) {
    values[index] += 1;
  }
}
