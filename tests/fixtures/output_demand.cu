// Offline-only fixture for engine/device_step/output_demand_physical_wbtest.mbt.
extern "C" __global__ void effect(int *state) { state[0] += 1; }
extern "C" __global__ void head(int *state) { state[1] += 1; }
extern "C" __global__ void sample(int *state) { state[2] += 1; }
