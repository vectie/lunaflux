///|
/// The exact separately materialized projection + epilogue, not a synthetic
/// standalone norm kernel. Uses the same input, weights, descriptors and
/// sentinel-protected output/KV buffers as the full-fusion differential probe.
fn ingress_cut_launches(
  functions : ArrayView[@device.Function],
  allocations : ArrayView[@device.Allocation],
  host : ArrayView[Bytes],
) -> Array[@device.OrderedKernelLaunch] raise {
  let arguments = fn(
    indices : ArrayView[Int],
  ) -> @device.KernelArguments raise {
    @device.KernelArguments::new(
      indices.map(index => {
        qwen3_region(
          allocations[index],
          host[index].length(),
          if index < 6 {
            4L
          } else {
            16L
          },
        )
      }),
    )
  }
  [
    @device.OrderedKernelLaunch::new(
      functions[2],
      @device.KernelLaunchDimensions::new(
        grid_x=512,
        grid_y=1,
        grid_z=1,
        block_x=256,
        block_y=1,
        block_z=1,
        shared_memory_bytes=8192,
      ),
      arguments([0, 6, 7, 8, 9, 15]),
    ),
    @device.OrderedKernelLaunch::new(
      functions[1],
      @device.KernelLaunchDimensions::new(
        grid_x=256,
        grid_y=32,
        grid_z=1,
        block_x=128,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      arguments([0, 1, 2, 3, 4, 5, 15, 10, 11, 12, 13, 14]),
    ),
  ]
}
