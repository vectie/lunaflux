///|
test "kernel launch dimensions admit one bounded geometry" {
  ignore(
    @device.KernelLaunchDimensions::new(
      grid_x=3,
      grid_y=2,
      grid_z=1,
      block_x=8,
      block_y=4,
      block_z=2,
      shared_memory_bytes=64,
    ),
  )
}

///|
test "kernel launch dimensions reject unsafe geometry" {
  let invalid : Array[(Int, Int, Int, Int, Int, Int, Int)] = [
    (0, 1, 1, 1, 1, 1, 0),
    (1, 65536, 1, 1, 1, 1, 0),
    (1, 1, 1, 1024, 2, 1, 0),
    (1, 1, 1, 1, 1, 1, -1),
    (1, 1, 1, 1, 1, 1, 98305),
  ]
  for dimensions in invalid {
    try
      @device.KernelLaunchDimensions::new(
        grid_x=dimensions.0,
        grid_y=dimensions.1,
        grid_z=dimensions.2,
        block_x=dimensions.3,
        block_y=dimensions.4,
        block_z=dimensions.5,
        shared_memory_bytes=dimensions.6,
      )
    catch {
      InvalidArgument(AotKernelLaunch) => ()
      error => fail("unexpected launch validation error: \{Repr(error)}")
    } noraise {
      _ => fail("unsafe launch dimensions were accepted")
    }
  }
}
