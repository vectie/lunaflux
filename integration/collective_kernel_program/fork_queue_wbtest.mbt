///|
test "physical independent branches replay and retire before storage release" {
  guard @env.get_env_var("LUNA_FORK_PROBE_ROOT") is Some(path) else { return }
  let root = @approved_fs.ApprovedRoot::open_absolute(path)
  defer root.close()
  let file = root.open_file(
    @approved_fs.ApprovedRelativeLocator::new("fork.cubin"),
  )
  defer file.close()
  let context = @device.open_context(0)
  defer context.close()
  let stream = context.create_stream()
  defer stream.close()
  let module = context.load_module(
    file.read_immutable_snapshot(maximum_bytes=1048576L),
  )
  defer module.close()
  let branch = module.load_function(b"fork_branch")
  defer branch.close()
  let join = module.load_function(b"fork_join")
  defer join.close()
  let buffers = Array::makei(4, _ => context.allocate(128L))
  defer (for buffer in buffers.rev() {
    buffer.close()
  })
  fn launch(
    f : @device.Function,
    allocations : ArrayView[@device.Allocation],
  ) -> @device.OrderedKernelLaunch raise {
    @device.OrderedKernelLaunch::new(
      f,
      @device.KernelLaunchDimensions::new(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        block_x=32,
        block_y=1,
        block_z=1,
        shared_memory_bytes=0,
      ),
      @device.KernelArguments::new(
        allocations.map(a => {
          @device.KernelArgument::device_region(
            a,
            offset=0L,
            byte_count=128L,
            alignment=4L,
          )
        }),
      ),
    )
  }
  let launches = [
    launch(branch, [buffers[0], buffers[1]]),
    launch(branch, [buffers[0], buffers[2]]),
    launch(join, [buffers[1], buffers[2], buffers[3]]),
  ]
  let forks = [@parallel.ForkJoinSpan::new(first=0, split=1, join=2)]
  for capture in [false, true] {
    let queue = PreparedQueue::new(context, stream, launches, forks, capture~)
    for iteration in 0..<32 {
      let input = FixedArray::make(128, b'\x00')
      for i in 0..<32 {
        input[i * 4] = iteration.to_byte()
      }
      buffers[0].copy_from_host(Bytes::from_array(input[:]), 0L, 0L, 128L)
      if capture {
        queue.launch_captured()
      } else {
        for i in 0..<3 {
          queue.enqueue(i)
        }
      }
      queue.record_completion()
      while !queue.poll_completion() {
        ()
      }
      let output = buffers[3].copy_to_host(0L, 128)
      for i in 0..<32 {
        assert_eq(output[i * 4], ((iteration + 1) * 2).to_byte())
        for b in 1..<4 {
          assert_eq(output[i * 4 + b], b'\x00')
        }
      }
      queue.reset()
    }
    queue.close()
    println(
      "physical_fork capture=\{capture} repetitions=32 exact=true released=true",
    )
  }
  let partial = PreparedQueue::new(context, stream, launches, forks)
  partial.enqueue(0)
  partial.abort()
  partial.close()
  println("physical_fork partial_abort_drained=true")
}
