///|
pub struct AudioSource {
  plan : @audio.AudioEncoder
  kernels : @lower.Source
}

///|
pub fn AudioSource::new(
  samples~ : Int,
  max_scratch_bytes~ : Int64,
) -> Self raise @plan.PlanError {
  let plan = @audio.AudioEncoder::new(samples~, max_scratch_bytes~)
  { plan, kernels: @lower.render(plan.graph), }
}

///|
fn region(
  a : @device.Allocation,
  offset : Int64,
  elements : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    a,
    offset~,
    byte_count=elements * 4L,
    alignment=4L,
  )
}

///|
/// Resolves exact checkpoint tensor ordinals against the existing completed
/// segmented upload. No second complete-model copy or per-request file I/O.
fn AudioSource::weight_arguments(
  self : Self,
  manifest : WeightManifest,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity : @spec.ModelIdentity,
) -> Map[String, @device.KernelArgument] raise {
  resolve_weight_arguments(
    self.plan.graph,
    AudioVae,
    manifest,
    weights,
    expected_identity,
  )
}

///|
fn resolve_weight_arguments(
  graph : @plan.Graph,
  component : @model_weights.MiniMaxH3WeightComponent,
  manifest : WeightManifest,
  weights : @materialize.SegmentedDeviceWeights,
  expected_identity : @spec.ModelIdentity,
) -> Map[String, @device.KernelArgument] raise {
  guard manifest.identity == expected_identity &&
    manifest.component == component else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let layout = weights.layout()
  guard layout.source_identity() == expected_identity &&
    layout.source_digest().as_hex() == manifest.digest &&
    layout.region_count() == manifest.tensors.length() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let tensors : Map[String, Int] = Map([])
  for i, tensor in manifest.tensors {
    tensors[tensor.name()] = i
  }
  let resolved : Map[String, @device.KernelArgument] = Map([])
  for value in graph.values {
    if value.origin is Weight(name) {
      guard tensors.get(name) is Some(ordinal) &&
        layout.region(ordinal) is Some(r) &&
        r.tensor_ordinal() == ordinal &&
        r.byte_count() == value.elements * 4L else {
        raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
      }
      let tensor = manifest.tensors[ordinal]
      guard r.source_arena_ordinal() == tensor.arena_index() &&
        r.source_offset().to_int64() == tensor.arena_offset() &&
        r.byte_count() == tensor.byte_length() else {
        raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
      }
      resolved[name] = region(
        weights.allocation(),
        r.device_offset(),
        value.elements,
      )
    }
  }
  resolved
}

///|
fn AudioSource::bind(
  self : Self,
  functions : ArrayView[@device.Function],
  weight_arguments : Map[String, @device.KernelArgument],
  input : @device.Allocation,
  input_offset : Int64,
  statistics : @device.Allocation,
  scratch : @device.Allocation,
) -> Array[@device.OrderedKernelLaunch] raise @device.DeviceError {
  let graph = self.plan.graph
  guard functions.length() == graph.nodes.length() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let values : Array[@device.KernelArgument] = []
  for i, value in graph.values {
    values.push(
      match value.origin {
        Temporary => region(scratch, graph.offsets[i], value.elements)
        Weight(name) =>
          match weight_arguments.get(name) {
            Some(arg) => arg
            None => raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
          }
        Input =>
          if i == 0 {
            region(input, input_offset, value.elements)
          } else if i == self.plan.means_value {
            region(statistics, 0L, 32L)
          } else if i == self.plan.deviations_value {
            region(statistics, 128L, 32L)
          } else {
            raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
          }
      },
    )
  }
  let launches : Array[@device.OrderedKernelLaunch] = []
  for i, node in graph.nodes {
    let arguments = node.inputs[:].map(id => values[id])
    while arguments.length() < 5 {
      arguments.push(arguments[0])
    }
    arguments.push(values[node.output])
    let (grid, threads) = self.kernels.grids[i]
    launches.push(
      @device.OrderedKernelLaunch::new(
        functions[i],
        @device.KernelLaunchDimensions::new(
          grid_x=grid,
          grid_y=1,
          grid_z=1,
          block_x=threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments),
      ),
    )
  }
  launches
}
