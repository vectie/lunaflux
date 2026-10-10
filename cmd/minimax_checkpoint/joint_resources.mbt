///|
// One explicit startup owner, not a global registry. Audio decoder modules
// have their own owner; only borrowed prediction/conditioning modules live here.
struct JointModules {
  context : @device.Context
  root : @approved_fs.ApprovedRoot
  modules : Array[@device.Module]
  functions : Array[@device.Function]
  mut loaded : Int64
  maximum : Int64
}

///|
fn JointModules::load(self : Self, name : String) -> @device.Module raise {
  let bytes = read(self.root, name + ".cubin", self.maximum - self.loaded)
  self.loaded += bytes.length().to_int64()
  let module = self.context.load_module(bytes)
  self.modules.push(module)
  module
}

///|
fn JointModules::function(
  self : Self,
  module : String,
  symbol : String,
) -> @device.Function raise {
  let function = self.load(module).load_function(@utf8.encode(symbol))
  self.functions.push(function)
  function
}

///|
fn JointModules::close(self : Self) -> Unit raise {
  while !self.functions.is_empty() {
    let index = self.functions.length() - 1
    self.functions[index].close()
    self.functions.remove(index) |> ignore
  }
  while !self.modules.is_empty() {
    let index = self.modules.length() - 1
    self.modules[index].close()
    self.modules.remove(index) |> ignore
  }
}

///|
fn inspect_joint_component(
  root : @approved_fs.ApprovedRoot,
  component_root : @approved_fs.ApprovedRoot,
  metadata : @h3.MiniMaxH3ModelMetadata,
  model : @joint.JointDiffusionPlan,
  component : @weights.MiniMaxH3WeightComponent,
  inventory : String,
  maximum : Int64,
) -> @checkpoint.ComponentCheckpoint raise {
  @checkpoint.ComponentCheckpoint::inspect(
    component_root,
    metadata,
    model,
    component,
    @inventory.CheckpointInventory::parse(
      @utf8.decode(read(root, inventory, 1048576L)),
      maximum_bytes=1048576,
      maximum_shards=64,
    ),
    maximum_component_bytes=maximum,
    maximum_device_bytes=maximum,
  )
}

///|
fn upload_joint_bytes(
  context : @device.Context,
  bytes : Bytes,
) -> @device.Allocation raise {
  let allocation = context.allocate(bytes.length().to_int64())
  allocation.copy_from_host(bytes, 0L, 0L, bytes.length().to_int64()) catch {
    error => {
      allocation.close()
      raise error
    }
  }
  allocation
}

///|
// Bracket initialization separately so its leases retire before the request.
fn pack_joint_noise(
  context : @device.Context,
  stream : @device.Stream,
  storage : @latent.LatentStorage,
  modules : JointModules,
) -> Unit raise {
  let pack = storage.boundary_launch(
    modules.function("video_pack", "lunaflux_dense_permutation_u32_v1"),
    boundary=PackVideoInput,
    block_size=256,
  )
  let queue = context.create_ordered_kernel_executor(
    stream,
    [pack],
    policy=OrderedKernelEagerOnly,
  )
  defer {
    queue.abort()
    queue.close()
  }
  queue.enqueue(0)
  queue.record_completion()
  queue.wait_completion()
}
