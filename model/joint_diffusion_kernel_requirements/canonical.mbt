///|
priv struct RequirementCanonicalWriter {
  hasher : @canonical_sha256.CanonicalSha256
  mut byte_count : Int
}

///|
fn RequirementCanonicalWriter::new() -> Self {
  { hasher: @canonical_sha256.CanonicalSha256::new(), byte_count: 0, }
}

///|
fn RequirementCanonicalWriter::reserve(
  self : Self,
  count : Int,
) -> Unit raise JointDiffusionKernelRequirementsError {
  if count < 0 || self.byte_count > 32768 - count {
    raise RequirementsEncodingTooLarge
  }
  self.byte_count += count
}

///|
fn RequirementCanonicalWriter::byte(
  self : Self,
  value : Byte,
) -> Unit raise JointDiffusionKernelRequirementsError {
  self.reserve(1)
  if !self.hasher.update_byte(value) {
    abort("finalized joint diffusion requirement hash")
  }
}

///|
fn RequirementCanonicalWriter::int(
  self : Self,
  value : Int,
) -> Unit raise JointDiffusionKernelRequirementsError {
  self.reserve(8)
  if !self.hasher.update_u64_le(Int64::from_int(value).reinterpret_as_uint64()) {
    abort("finalized joint diffusion requirement hash")
  }
}

///|
fn RequirementCanonicalWriter::text(
  self : Self,
  value : String,
) -> Unit raise JointDiffusionKernelRequirementsError {
  let bytes = @utf8.encode(value)
  self.int(bytes.length())
  self.reserve(bytes.length())
  if !self.hasher.update_bytes(bytes) {
    abort("finalized joint diffusion requirement hash")
  }
}

///|
fn encode_operation(
  writer : RequirementCanonicalWriter,
  operation : JointDiffusionSemanticOperation,
) -> Unit raise JointDiffusionKernelRequirementsError {
  match operation {
    TokenTextConditioning(dimension) => {
      writer.byte(1)
      writer.int(dimension)
    }
    OrderedKeyframeConditioning(maximum, channels) => {
      writer.byte(2)
      writer.int(maximum)
      writer.int(channels)
    }
    OrderedReferenceConditioning(images, videos, audio, total, audio_only) => {
      writer.byte(3)
      writer.int(images)
      writer.int(videos)
      writer.int(audio)
      writer.int(total)
      writer.byte(if audio_only { 1 } else { 0 })
    }
    VideoLatentProjection(channels, temporal, height, width, hidden) => {
      writer.byte(4)
      writer.int(channels)
      writer.int(temporal)
      writer.int(height)
      writer.int(width)
      writer.int(hidden)
    }
    AudioLatentProjection(channels, hidden) => {
      writer.byte(5)
      writer.int(channels)
      writer.int(hidden)
    }
    JointTransformerAttention(layers, refiners, heads, head_dim, hidden) => {
      writer.byte(6)
      writer.int(layers)
      writer.int(refiners)
      writer.int(heads)
      writer.int(head_dim)
      writer.int(hidden)
    }
    JointTransformerMlp(layers, refiners, hidden, intermediate) => {
      writer.byte(7)
      writer.int(layers)
      writer.int(refiners)
      writer.int(hidden)
      writer.int(intermediate)
    }
    ShiftedFlowStep(latent, numerator, denominator, timesteps) => {
      writer.byte(8)
      writer.byte(
        match latent {
          VideoLatent => 1
          AudioLatent => 2
        },
      )
      writer.int(numerator)
      writer.int(denominator)
      writer.int(timesteps)
    }
    VideoVaeLatentDestandardization(channels, storage, layout) => {
      writer.byte(9)
      writer.int(channels)
      writer.byte(
        match storage {
          Float32InputScaleMeanOutput => 1
        },
      )
      writer.byte(
        match layout {
          BatchChannelFrameHeightWidth => 1
        },
      )
    }
    VideoVaeDecoderBody(channels, compression, frames, latents) => {
      writer.byte(22)
      writer.int(channels)
      writer.int(compression)
      writer.int(frames)
      writer.int(latents)
    }
    AudioVaeLatentDestandardization(channels, batch, storage, layout) => {
      writer.byte(10)
      writer.int(channels)
      writer.int(batch)
      writer.byte(
        match storage {
          Float32InputScaleMeanOutput => 1
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderInputProjection(
      batch,
      input,
      output,
      kernel,
      storage,
      layout
    ) => {
      writer.byte(23)
      writer.int(batch)
      writer.int(input)
      writer.int(output)
      writer.int(kernel)
      writer.byte(
        match storage {
          Float32InputWeightBiasOutput => 1
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderConvPre(
      batch,
      input,
      output,
      kernel,
      padding,
      stride,
      storage,
      axes,
      layout
    ) => {
      writer.byte(24)
      writer.int(batch)
      writer.int(input)
      writer.int(output)
      writer.int(kernel)
      writer.int(padding)
      writer.int(stride)
      writer.byte(
        match storage {
          Float32InputWeightGWeightVBiasWorkspaceOutput => 1
        },
      )
      writer.byte(
        match axes {
          PerOutputChannelInputAndKernel => 1
          PerInputChannelOutputAndKernel => 2
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderBody(decoder, latent_rate, sample_rate, outputs) => {
      writer.byte(26)
      writer.int(decoder)
      writer.int(latent_rate)
      writer.int(sample_rate)
      writer.int(outputs)
    }
    AudioVaeDecoderFirstUpsample(
      batch,
      input,
      output,
      kernel,
      stride,
      padding,
      storage,
      axes,
      layout
    ) => {
      writer.byte(25)
      writer.int(batch)
      writer.int(input)
      writer.int(output)
      writer.int(kernel)
      writer.int(stride)
      writer.int(padding)
      writer.byte(
        match storage {
          Float32InputWeightGWeightVBiasWorkspaceOutput => 1
        },
      )
      writer.byte(
        match axes {
          PerInputChannelOutputAndKernel => 2
          PerOutputChannelInputAndKernel => 1
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderFirstAmpActivation(
      batch,
      channels,
      ratio,
      kernel,
      storage,
      layout
    ) => {
      writer.byte(27)
      writer.int(batch)
      writer.int(channels)
      writer.int(ratio)
      writer.int(kernel)
      writer.byte(
        match storage {
          Float32InputAlphaBetaUpsampleFilterDownsampleFilterWorkspacesOutput =>
            1
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderFirstAmpStage(
      batch,
      channels,
      blocks,
      pairs,
      storage,
      layout
    ) => {
      writer.byte(28)
      writer.int(batch)
      writer.int(channels)
      writer.int(blocks)
      writer.int(pairs)
      writer.byte(
        match storage {
          Float32OriginalPrecomputedActivationPackedParametersWorkspaceOutput =>
            1
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderUpsampleTransition(
      stage,
      batch,
      input,
      output,
      kernel,
      stride,
      padding,
      storage,
      axes,
      layout
    ) => {
      writer.byte(29)
      writer.int(stage)
      writer.int(batch)
      writer.int(input)
      writer.int(output)
      writer.int(kernel)
      writer.int(stride)
      writer.int(padding)
      writer.byte(
        match storage {
          Float32InputWeightGWeightVBiasWorkspaceOutput => 1
        },
      )
      writer.byte(
        match axes {
          PerOutputChannelInputAndKernel => 1
          PerInputChannelOutputAndKernel => 2
        },
      )
      writer.byte(
        match layout {
          BatchChannelTime => 1
        },
      )
    }
    AudioVaeDecoderAmpActivation(
      stage,
      batch,
      channels,
      ratio,
      kernel,
      storage,
      layout
    ) =>
      encode_audio_amp_activation(
        writer, stage, batch, channels, ratio, kernel, storage, layout,
      )
    AudioVaeDecoderAmpStage(
      stage,
      batch,
      channels,
      blocks,
      pairs,
      schedule,
      storage,
      layout
    ) =>
      encode_audio_amp_stage(
        writer, stage, batch, channels, blocks, pairs, schedule, storage, layout,
      )
    VideoOutputProjection(channels, temporal, height, width, hidden) => {
      writer.byte(11)
      writer.int(channels)
      writer.int(temporal)
      writer.int(height)
      writer.int(width)
      writer.int(hidden)
    }
    AudioOutputProjection(channels, hidden) => {
      writer.byte(12)
      writer.int(channels)
      writer.int(hidden)
    }
    VideoFinalRmsNorm(hidden, numerator, denominator) => {
      writer.byte(13)
      writer.int(hidden)
      writer.int(numerator)
      writer.int(denominator)
    }
    AudioFinalRmsNorm(hidden, numerator, denominator) => {
      writer.byte(14)
      writer.int(hidden)
      writer.int(numerator)
      writer.int(denominator)
    }
    VideoFinalAdaLnF32Boundary(hidden) => {
      writer.byte(15)
      writer.int(hidden)
    }
    AudioFinalAdaLnF32Boundary(hidden) => {
      writer.byte(16)
      writer.int(hidden)
    }
    FinalAdaLnParameterRow(input_width, hidden) => {
      writer.byte(17)
      writer.int(input_width)
      writer.int(hidden)
    }
    PackedModalityAdaLnRows(hidden) => {
      writer.byte(18)
      writer.int(hidden)
    }
    TimestepFrequencyRows(width) => {
      writer.byte(19)
      writer.int(width)
    }
    TimestepEmbeddingMlpRows(input, hidden, output) => {
      writer.byte(20)
      writer.int(input)
      writer.int(hidden)
      writer.int(output)
    }
    DistinctTimestepAdaLnParameterRows(input, output) => {
      writer.byte(21)
      writer.int(input)
      writer.int(output)
    }
  }
}

///|
fn capability_tag(capability : JointDiffusionAotCapability) -> Byte {
  match capability {
    TokenTextConditioningAot => 1
    OrderedKeyframeConditioningAot => 2
    OrderedReferenceConditioningAot => 3
    VideoLatentProjectionAot => 4
    AudioLatentProjectionAot => 5
    JointTransformerAttentionAot => 6
    JointTransformerMlpAot => 7
    ShiftedFlowStepAot => 8
    VideoVaeLatentDestandardizationAot => 9
    VideoVaeDecoderBodyAot => 22
    AudioVaeLatentDestandardizationAot => 10
    AudioVaeDecoderInputProjectionAot => 23
    AudioVaeDecoderConvPreAot => 24
    AudioVaeDecoderFirstUpsampleAot => 25
    AudioVaeDecoderFirstAmpActivationAot => 27
    AudioVaeDecoderFirstAmpStageAot => 28
    AudioVaeDecoderUpsampleTransitionAot => 29
    AudioVaeDecoderAmpActivationAot => 30
    AudioVaeDecoderAmpStageAot => 31
    AudioVaeDecoderBodyAot => 26
    VideoOutputProjectionAot => 11
    AudioOutputProjectionAot => 12
    VideoFinalRmsNormAot => 13
    AudioFinalRmsNormAot => 14
    VideoFinalAdaLnF32BoundaryAot => 15
    AudioFinalAdaLnF32BoundaryAot => 16
    FinalAdaLnParameterRowAot => 17
    PackedModalityAdaLnRowsAot => 18
    TimestepFrequencyRowsAot => 19
    TimestepEmbeddingMlpRowsAot => 20
    DistinctTimestepAdaLnParameterRowsAot => 21
  }
}
