///|
fn String::substitute(self : String, old : String, new : String) -> String {
  self.split(old).join(new)
}

///|
async test "offline control arithmetic probe retains the actual renderer" {
  let plan = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=144,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    mix_law=TransposedF32ResidualThenBranch,
    control_law=F32ProjectionThenNormalize,
  )
  let production = @hyper_connection_execution_cuda_source.hyper_connection_source(
    plan,
    symbol_prefix="hc_probe",
  )
  let base = @utf8.decode(production.source_bytes())
  let header = arithmetic_capture(base, "production")
  let sources : Array[String] = [header]
  let refined = @hyper_connection_execution_cuda_source.hyper_connection_source(
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=144,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      mix_law=TransposedF32ResidualThenBranch,
      control_law=F32ProjectionThenNormalize,
      control_square_law=CompensatedFloat32,
      control_dot_law=StripedTreeFloat32(partitions=32),
    ),
    symbol_prefix="hc_refined",
  )
  sources.push(
    arithmetic_capture(@utf8.decode(refined.source_bytes()), "refined"),
  )
  let nonlinear = @hyper_connection_execution_cuda_source.hyper_connection_source(
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=144,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      mix_law=TransposedF32ResidualThenBranch,
      control_law=F32ProjectionThenNormalize,
      control_square_law=CompensatedFloat32,
      control_dot_law=StripedTreeFloat32(partitions=32),
      control_vector_affine_law=FusedMultiplyAdd,
      control_sigmoid_law=DirectReciprocal,
    ),
    symbol_prefix="hc_nonlinear",
  )
  sources.push(
    arithmetic_capture(@utf8.decode(nonlinear.source_bytes()), "nonlinear"),
  )
  let vector = @hyper_connection_execution_cuda_source.hyper_connection_source(
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=144,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      mix_law=TransposedF32ResidualThenBranch,
      control_law=F32ProjectionThenNormalize,
      control_square_law=VectorizedTreeFloat32(partitions=512, vector_width=4),
      control_dot_law=StripedTreeFloat32(partitions=32),
      control_vector_affine_law=FusedMultiplyAdd,
      control_sigmoid_law=DirectReciprocal,
      control_root_law=ReciprocalSqrtFloat32,
      normalization_square_law=VectorizedTreeFloat32(
        partitions=512,
        vector_width=4,
      ),
    ),
    symbol_prefix="hc_vector_rms",
  )
  assert_eq(vector.normalization_threads(), 512)
  sources.push(
    arithmetic_capture(@utf8.decode(vector.source_bytes()), "vector_rms"),
  )
  let hierarchical = @hyper_connection_execution_cuda_source.hyper_connection_source(
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=144,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      mix_law=TransposedF32ResidualThenBranch,
      control_law=F32ProjectionThenNormalize,
      control_square_law=VectorizedTreeFloat32(partitions=512, vector_width=4),
      control_dot_law=GroupedVectorFloat32(
        groups=18,
        partitions=128,
        vector_width=2,
        publication_partitions=4,
        multiply_add=FusedMultiplyAdd,
      ),
      control_vector_affine_law=FusedMultiplyAdd,
      control_sigmoid_law=DirectReciprocal,
      control_root_law=ReciprocalSqrtFloat32,
      normalization_square_law=VectorizedTreeFloat32(
        partitions=512,
        vector_width=4,
      ),
    ),
    symbol_prefix="hc_hierarchical",
  )
  assert_eq(hierarchical.normalization_threads(), 512)
  sources.push(
    arithmetic_capture(
      @utf8.decode(hierarchical.source_bytes()),
      "hierarchical",
    ),
  )
  for
    route in [
      "refined_direct", "refined_affine_fma", "refined_direct_affine_fma", "refined_rsqrt",
    ] {
    let source = @utf8.decode(refined.source_bytes()).substitute(
      "hc_refined_",
      "hc_" + route + "_",
    )
    sources.push(arithmetic_capture(source, route))
  }
  for
    route in [
      "literal_mix", "literal_mix_direct", "literal_mix_affine_fma", "literal_mix_direct_affine_fma",
      "literal_inverse", "striped32", "striped32_fma", "striped32_literal_inverse",
      "striped32_fma_literal_inverse", "compensated_square", "square256", "striped32_fma_compensated_square",
      "striped32_fma_square256",
    ] {
    let raw = @control.staged_function_and_collapse_bf16_f32_source(
      stream_width=4,
      hidden_width=4096,
      maximum_rows=144,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      function_symbol="hc_" + route + "_function",
      control_law=F32ProjectionThenNormalize,
    )
    let text = @utf8.decode(raw.source_bytes()).substitute(
      "lf_hc_stage_sigmoid",
      "hc_" + route + "_sigmoid",
    )
    assert_false(text.contains("lf_hc_stage_sigmoid"))
    assert_eq(text.split("hc_" + route + "_sigmoid").count(), 4)
    sources.push(arithmetic_capture(text, route))
  }
  if @env.get_env_var("LUNAFLUX_HYPER_ARITHMETIC_SOURCE") is Some(path) {
    @fs.write_file(
      path,
      @utf8.encode(sources.join("\n")),
      create_mode=CreateNew,
    )
  }
}

///|
/// Diagnostic-only source interventions; no literal reference enters serving.
fn arithmetic_capture(source : String, route : String) -> String raise {
  let parameters = "const float *scale, float *post"
  assert_eq(source.split(parameters).count(), 2)
  let text = source.substitute(
    parameters, "const float *scale, const float *literal_mixes, const float *literal_inverse, float *measured_mixes, float *measured_inverse, float *post",
  )
  let text = if route.contains("literal_inverse") {
    text
    .substitute("const float inverse = ", "const float computed_inverse = ")
    .substitute(
      "if (!isfinite(inverse)) return; float mixes[24];", "const float inverse = literal_inverse[row]; if (!isfinite(inverse)) return; float mixes[24];",
    )
  } else {
    text
  }
  let text = if route.has_prefix("striped32") {
    assert_eq(
      text.split("dot = __fadd_rn(dot, __fmul_rn(operand, weight));").count(),
      2,
    )
    text
    .substitute(
      "float dot = 0.0f;", "float dot = 0.0f; float partial[32] = {};",
    )
    .substitute(
      "dot = __fadd_rn(dot, __fmul_rn(operand, weight));",
      if route.contains("fma") {
        "partial[column % 32] = __fmaf_rn(operand, weight, partial[column % 32]);"
      } else {
        "partial[column % 32] = __fadd_rn(partial[column % 32], __fmul_rn(operand, weight));"
      },
    )
    .substitute(
      "} mixes[output] =", "} for (int span = 16; span > 0; span >>= 1) for (int p = 0; p < span; ++p) partial[p] = __fadd_rn(partial[p], partial[p + span]); dot = partial[0]; mixes[output] =",
    )
  } else {
    text
  }
  let text = if route.contains("compensated_square") {
    text
    .substitute(
      "float square_sum = 0.0f;", "float square_sum = 0.0f, square_correction = 0.0f;",
    )
    .substitute(
      "square_sum = __fadd_rn(square_sum, __fmul_rn(x, x));", "const float product = __fmul_rn(x, x), next = __fadd_rn(square_sum, product); const float lost = fabsf(square_sum) >= fabsf(product) ? __fadd_rn(__fsub_rn(square_sum, next), product) : __fadd_rn(__fsub_rn(product, next), square_sum); square_correction = __fadd_rn(square_correction, lost); square_sum = next;",
    )
    .substitute(
      "const float inverse =", "square_sum = __fadd_rn(square_sum, square_correction); const float inverse =",
    )
  } else if route.contains("square256") {
    text
    .substitute(
      "float square_sum = 0.0f;", "float square_sum = 0.0f; float square_partial[256] = {};",
    )
    .substitute(
      "square_sum = __fadd_rn(square_sum, __fmul_rn(x, x));", "square_partial[column % 256] = __fadd_rn(square_partial[column % 256], __fmul_rn(x, x));",
    )
    .substitute(
      "const float inverse =", "for (int span = 128; span > 0; span >>= 1) for (int p = 0; p < span; ++p) square_partial[p] = __fadd_rn(square_partial[p], square_partial[p + span]); square_sum = square_partial[0]; const float inverse =",
    )
  } else {
    text
  }
  let text = if route.has_prefix("literal_mix") {
    text.substitute(
      "mixes[output] = __fmul_rn(dot, inverse);", "mixes[output] = literal_mixes[(unsigned long long)row * 24ULL + output];",
    )
  } else {
    text
  }
  let text = control_nonlinear_counterfactual(text, route)
  text
  .substitute(
    "float mixes[24];", "measured_inverse[row] = inverse; float mixes[24];",
  )
  .substitute(
    "for (int32_t i = 0; i < 24; ++i) if (!isfinite(base[i]))", "for (int i = 0; i < 24; ++i) measured_mixes[(unsigned long long)row * 24ULL + i] = mixes[i]; for (int32_t i = 0; i < 24; ++i) if (!isfinite(base[i]))",
  )
}
