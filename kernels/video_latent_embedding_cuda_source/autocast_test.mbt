///|
test "video transformer storage obeys explicit FP16 law without changing F32 residuals" {
  let source = @utf8.decode(
    @video_latent_embedding_cuda_source.attention_stem_source(
      temporal=1,
      height=2,
      width=3,
      heads=2,
      head_dimension=6,
      rotary_dimension=6,
      register_rows=4,
      prefix="probe",
      autocast=Float16,
    ),
  )
  assert_true(source.contains("#include <cuda_fp16.h>"))
  assert_true(source.contains("__half2float"))
  assert_false(source.contains("__nv_bfloat16"))
  assert_true(source.contains("__half2float(__float2half_rn(__fdiv_rn("))
  let tail = @utf8.decode(
    @video_latent_embedding_cuda_source.transformer_tail_source(
      rows=11,
      hidden=12,
      heads=2,
      head_dimension=6,
      intermediate=24,
      prefix="probe",
      autocast=Float16,
    ),
  )
  assert_true(
    tail.contains("const __half *delta,const float *scale,float *output"),
  )
  assert_false(tail.contains("__float2bfloat16_rn"))
  if @env.get_env_var("LUNA_EXPORT_VIDEO_FP16") == Some("1") {
    println("LUNA_VIDEO_FP16_BEGIN")
    println(source)
    println(tail)
    println(
      @utf8.decode(
        @video_latent_embedding_cuda_source.render(
          batch=1,
          positions=6,
          input_channels=3,
          hidden_channels=12,
          register_rows=4,
          symbol="embed",
          output_rounding=Some(Float16),
        ).source,
      ),
    )
    println(
      @utf8.decode(
        @projection.f32_pointwise_conv1d_source(
          batch_count=1,
          input_channels=3,
          output_channels=3,
          time_count=6,
          function_symbol="post_quant",
          autocast=Some(Float16),
        ).source_bytes(),
      ),
    )
    println("LUNA_VIDEO_FP16_END")
  }
}

///|
test "embedding keeps F32 module arithmetic then publishes FP16 before F32 register concatenation" {
  let source = @utf8.decode(
    @video_latent_embedding_cuda_source.render(
      batch=1,
      positions=2,
      input_channels=3,
      hidden_channels=5,
      register_rows=4,
      symbol="embed",
      output_rounding=Some(Float16),
    ).source,
  )
  assert_true(source.contains("float sum = 0.0f"))
  assert_true(
    source.contains(
      "__half2float(__float2half_rn(__fadd_rn(sum, bias[channel])))",
    ),
  )
  assert_true(source.contains("const float *registers, float *output"))
}
