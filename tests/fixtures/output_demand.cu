// Offline-only fixture for engine/device_step/output_demand_physical_wbtest.mbt.
extern "C" __global__ void effect(int *state) { state[0] += 1; }
extern "C" __global__ void head(int *state) { state[1] += 1; }
extern "C" __global__ void sample(int *state) { state[2] += 1; }

extern "C" __global__ void observe_output_rows(
    const int *original, const int *projected, const int *ends, int *observed) {
  observed[0] = original[2];
  observed[1] = original[3];
  observed[2] = original[4];
  observed[3] = projected[2];
  for (int row = 0; row <= 4; ++row)
    observed[4 + row] = row <= projected[2] ? ends[row] : -7;
}
