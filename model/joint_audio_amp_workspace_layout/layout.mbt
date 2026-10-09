///|
fn expected_stages() -> Array[JointAudioAmpExecutionStage] {
  let stages : Array[JointAudioAmpExecutionStage] = []
  for block in 0..<3 {
    for pair in 0..<3 {
      let activation = pair * 2
      if !(block == 0 && pair == 0) {
        stages.push(Activation(block, activation, Upsample))
        stages.push(Activation(block, activation, SnakeBeta))
        stages.push(Activation(block, activation, Downsample))
      }
      stages.push(Convolution(block, pair, 0, Normalize))
      stages.push(Convolution(block, pair, 0, Convolve))
      stages.push(Activation(block, activation + 1, Upsample))
      stages.push(Activation(block, activation + 1, SnakeBeta))
      stages.push(Activation(block, activation + 1, Downsample))
      stages.push(Convolution(block, pair, 1, Normalize))
      stages.push(Convolution(block, pair, 1, Convolve))
      stages.push(Residual(block, pair))
    }
  }
  stages.push(ParallelAverage)
  stages
}

///|
pub fn validate_first_stage_amp_execution_stages(
  stages : ArrayView[JointAudioAmpExecutionStage],
) -> Unit raise JointAudioAmpWorkspaceLayoutError {
  let expected = expected_stages()
  if stages.length() != 97 {
    raise InvalidStageCount
  }
  for index, stage in stages {
    if stage != expected[index] {
      raise InvalidStageOrder(index)
    }
  }
}

///|
/// Validates the reusable 30+33+33+1 AMP composition when activation zero is
/// supplied as an independently authenticated input.
pub fn validate_amp_execution_stages(
  stages : ArrayView[JointAudioAmpExecutionStage],
) -> Unit raise JointAudioAmpWorkspaceLayoutError {
  validate_first_stage_amp_execution_stages(stages)
}

///|
fn checked_f32_bytes(
  elements : Int64,
) -> Int64 raise JointAudioAmpWorkspaceLayoutError {
  if elements <= 0L || elements > 9223372036854775807L / 4L {
    raise SizeOverflow
  }
  elements * 4L
}

///|
fn aligned_end(
  offset : Int64,
  bytes : Int64,
) -> Int64 raise JointAudioAmpWorkspaceLayoutError {
  if offset < 0L || bytes <= 0L || offset > 9223372036854775807L - bytes {
    raise SizeOverflow
  }
  let end = offset + bytes
  if end > 9223372036854775807L - 255L {
    raise SizeOverflow
  }
  (end + 255L) / 256L * 256L
}

///|
pub fn first_stage_amp_workspace_layout(
  element_count~ : Int64,
  intermediate_element_count~ : Int64,
  normalized_weight_element_count~ : Int64,
) -> JointAudioAmpWorkspaceLayout raise JointAudioAmpWorkspaceLayoutError {
  if element_count <= 0L ||
    element_count > 9223372036854775807L / 2L ||
    intermediate_element_count != element_count * 2L {
    raise InvalidGeometry
  }
  let stages = expected_stages()
  validate_first_stage_amp_execution_stages(stages)
  let value_bytes = checked_f32_bytes(element_count)
  let intermediate_bytes = checked_f32_bytes(intermediate_element_count)
  let weight_bytes = checked_f32_bytes(normalized_weight_element_count)
  let regions : Array[JointAudioAmpWorkspaceRegion] = []
  regions.push({
    role: OriginalUpsamplerOutput,
    storage_class: ExternalInput,
    byte_offset: -1L,
    byte_count: value_bytes,
    live_start: 0,
    live_end: 95,
  })
  regions.push({
    role: PrecomputedBlock0Activation0,
    storage_class: ExternalInput,
    byte_offset: -1L,
    byte_count: value_bytes,
    live_start: 0,
    live_end: 1,
  })
  let mut offset = 0L
  let workspace_roles : Array[(JointAudioAmpWorkspaceRole, Int64, Int, Int)] = [
    (ActivationUpsampleWorkspace, intermediate_bytes, 2, 93),
    (ActivationValueWorkspace, intermediate_bytes, 3, 94),
    (ActivationOutputWorkspace, value_bytes, 5, 95),
    (NormalizedWeightWorkspace, weight_bytes, 0, 94),
    (ConvolutionOutputWorkspace, value_bytes, 1, 95),
    (ResidualPingWorkspace, value_bytes, 0, 95),
    (ResidualPongWorkspace, value_bytes, 7, 95),
    (ParallelBlockOutput(0), value_bytes, 29, 96),
    (ParallelBlockOutput(1), value_bytes, 62, 96),
    (ParallelBlockOutput(2), value_bytes, 95, 96),
  ]
  for item in workspace_roles {
    let (role, bytes, live_start, live_end) = item
    regions.push({
      role,
      storage_class: Workspace,
      byte_offset: offset,
      byte_count: bytes,
      live_start,
      live_end,
    })
    offset = aligned_end(offset, bytes)
  }
  regions.push({
    role: FinalOutput,
    storage_class: ExternalOutput,
    byte_offset: -1L,
    byte_count: value_bytes,
    live_start: 96,
    live_end: 96,
  })
  for left_index, left in regions {
    if left.live_start < 0 ||
      left.live_end < left.live_start ||
      left.live_end >= 97 {
      raise InvalidRegion(left_index)
    }
    if left.storage_class == Workspace {
      if left.byte_offset < 0L || left.byte_offset % 256L != 0L {
        raise InvalidRegion(left_index)
      }
      for right_index in (left_index + 1)..<regions.length() {
        let right = regions[right_index]
        if right.storage_class == Workspace &&
          left.live_start <= right.live_end &&
          right.live_start <= left.live_end {
          let left_end = left.byte_offset + left.byte_count
          let right_end = right.byte_offset + right.byte_count
          if left.byte_offset < right_end && right.byte_offset < left_end {
            raise OverlappingLiveRegions(left_index, right_index)
          }
        }
      }
    } else if left.byte_offset != -1L {
      raise InvalidRegion(left_index)
    }
  }
  let canonical = StringBuilder()
  canonical.write_string(
    "lunaflux.joint-audio-amp-workspace-layout.v1|dtype=f32|alignment=256|stages=30+33+33+1",
  )
  for index, stage in stages {
    canonical <+ "|s\{index}:\{stage_tag(stage)}"
  }
  for index, region in regions {
    canonical <+
      "|r\{index}:\{role_tag(region.role)}:\{storage_tag(region.storage_class)}:\{region.byte_offset}:\{region.byte_count}:\{region.live_start}:\{region.live_end}"
  }
  {
    stages: ReadOnlyArray::from_array(stages),
    regions: ReadOnlyArray::from_array(regions),
    workspace_bytes: offset,
    digest: {
      hex: @crypto.bytes_to_hex_string(
        @crypto.sha256(@utf8.encode(canonical.to_string())),
      ),
    },
  }
}

///|
/// Builds the reusable workspace/liveness plan for one three-way AMP stage.
pub fn amp_workspace_layout(
  element_count~ : Int64,
  intermediate_element_count~ : Int64,
  normalized_weight_element_count~ : Int64,
) -> JointAudioAmpWorkspaceLayout raise JointAudioAmpWorkspaceLayoutError {
  first_stage_amp_workspace_layout(
    element_count~,
    intermediate_element_count~,
    normalized_weight_element_count~,
  )
}

///|
fn stage_tag(stage : JointAudioAmpExecutionStage) -> String {
  match stage {
    Activation(block, activation, phase) =>
      "a:\{block}:\{activation}:\{match phase { Upsample => 0; SnakeBeta => 1; Downsample => 2 }}"
    Convolution(block, pair, side, phase) =>
      "c:\{block}:\{pair}:\{side}:\{match phase { Normalize => 0; Convolve => 1 }}"
    Residual(block, pair) => "r:\{block}:\{pair}"
    ParallelAverage => "avg"
  }
}

///|
fn role_tag(role : JointAudioAmpWorkspaceRole) -> String {
  match role {
    OriginalUpsamplerOutput => "original"
    PrecomputedBlock0Activation0 => "activation0"
    ActivationUpsampleWorkspace => "activation-upsample"
    ActivationValueWorkspace => "activation-value"
    ActivationOutputWorkspace => "activation-output"
    NormalizedWeightWorkspace => "normalized-weight"
    ConvolutionOutputWorkspace => "convolution-output"
    ResidualPingWorkspace => "residual-ping"
    ResidualPongWorkspace => "residual-pong"
    ParallelBlockOutput(block) => "block-output:\{block}"
    FinalOutput => "final-output"
  }
}

///|
fn storage_tag(storage : JointAudioAmpStorageClass) -> Int {
  match storage {
    ExternalInput => 0
    Workspace => 1
    ExternalOutput => 2
  }
}
