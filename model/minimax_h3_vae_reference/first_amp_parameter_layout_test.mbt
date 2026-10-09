///|
test "all seven AMP stages derive complete nonoverlapping parameter layouts" {
  for stage in 0..<7 {
    let layout = @minimax_h3_vae_reference.bind_audio_decoder_amp_parameter_layout(
      stage_index=stage,
    )
    assert_eq(layout.regions().length(), 126)
    assert_eq(
      layout.regions()[0].tensor_name(),
      "decoder.resblocks.\{stage * 3}.activations.0.act.alpha",
    )
    assert_eq(
      layout.regions()[125].tensor_name(),
      "decoder.resblocks.\{stage * 3 + 2}.convs2.2.weight_v",
    )
    for i in 1..<layout.regions().length() {
      let previous = layout.regions()[i - 1]
      assert_true(
        layout.regions()[i].byte_offset() >=
        previous.byte_offset() + previous.byte_count(),
      )
    }
  }
  for stage in [-1, 7] {
    try
      @minimax_h3_vae_reference.bind_audio_decoder_amp_parameter_layout(
        stage_index=stage,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid AMP stage accepted")
    }
  }
}

///|
test "official first AMP packed layout authenticates all 126 regions" {
  let layout = try! @minimax_h3_vae_reference.bind_first_amp_parameter_layout()
  assert_eq(layout.regions().length(), 126)
  assert_true(layout.total_arena_bytes() > 0L)
  assert_eq(layout.regions()[0].role(), ActivationAlpha(0, 0))
  assert_eq(
    layout.regions()[0].tensor_name(),
    "decoder.resblocks.0.activations.0.act.alpha",
  )
  assert_eq(layout.regions()[125].role(), ConvolutionWeightV(2, 2, 1))
  assert_eq(
    layout.regions()[125].tensor_name(),
    "decoder.resblocks.2.convs2.2.weight_v",
  )
  for index in 1..<layout.regions().length() {
    let previous = layout.regions()[index - 1]
    let current = layout.regions()[index]
    assert_true(
      current.byte_offset() >= previous.byte_offset() + previous.byte_count(),
    )
    assert_eq(current.byte_offset() % 64L, 0L)
  }
  inspect(layout.total_arena_bytes(), content="132270336")
  inspect(
    layout.digest().as_hex(),
    content="607316676ae22c6134596ecbe52a94acf4f5f86b43d960ed95039769bc8f8e48",
  )
}

///|
test "official second AMP packed layout authenticates resblocks three through five" {
  let layout = try! @minimax_h3_vae_reference.bind_second_amp_parameter_layout()
  assert_eq(layout.regions().length(), 126)
  assert_eq(layout.regions()[0].role(), ActivationAlpha(0, 0))
  assert_eq(
    layout.regions()[0].tensor_name(),
    "decoder.resblocks.3.activations.0.act.alpha",
  )
  assert_eq(layout.regions()[125].role(), ConvolutionWeightV(2, 2, 1))
  assert_eq(
    layout.regions()[125].tensor_name(),
    "decoder.resblocks.5.convs2.2.weight_v",
  )
  inspect(
    layout.total_arena_bytes(),
    content=(
      #|33106176
    ),
  )
  inspect(
    layout.digest().as_hex(),
    content=(
      #|41b5b51edac3201fca3de8bef188cc0fb66503d03115797da1f397827f78d0ba
    ),
  )
}
