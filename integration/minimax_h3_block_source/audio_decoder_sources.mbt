///|
/// One resolved AudioVAE module set. Component bootstrap and full media source
/// export share this construction; neither invents a second decoder geometry.
pub struct AudioDecoderSources {
  frontend : AudioFrontendSource
  upsample : ReadOnlyArray[AudioUpsampleSource]
  amp : ReadOnlyArray[AudioAmpSource]
  tail : AudioTailSource
}

///|
pub fn build_audio_decoder(
  transitions : ArrayView[
    @joint_diffusion_vae_reference.JointVaeAudioDecoderUpsampleTransition,
  ],
) -> AudioDecoderSources raise {
  guard transitions.length() == 7 else {
    fail("expected seven AudioVAE transitions")
  }
  let first = transitions[0]
  let front = build_audio_frontend(
    batch=first.decoder_batch_count(),
    time=first.input_time(),
    function_prefix="h3_audio_front",
  )
  let upsample = []
  let amp = []
  for i, t in transitions {
    guard t.decoder_batch_count() == first.decoder_batch_count() &&
      (
        i == 0 ||
        (
          t.input_time() == transitions[i - 1].output_time() &&
          t.input_channels() == transitions[i - 1].output_channels()
        )
      ) else {
      fail("disconnected AudioVAE transitions")
    }
    upsample.push(
      build_audio_upsample(
        batch=t.decoder_batch_count(),
        input_channels=t.input_channels(),
        output_channels=t.output_channels(),
        input_time=t.input_time(),
        kernel=t.kernel_size(),
        stride=t.stride(),
        padding=t.padding(),
        function_prefix="h3_audio_up_\{i}",
      ),
    )
    amp.push(
      build_audio_amp(
        batch=t.decoder_batch_count(),
        channels=t.output_channels(),
        time=t.output_time(),
        function_prefix="h3_audio_amp_\{i}",
      ),
    )
  }
  let last = transitions[6]
  {
    frontend: front,
    upsample: ReadOnlyArray::from_array(upsample),
    amp: ReadOnlyArray::from_array(amp),
    tail: build_audio_tail(
      batch=last.decoder_batch_count(),
      channels=last.output_channels(),
      time=last.output_time(),
      function_prefix="h3_audio_tail",
    ),
  }
}
