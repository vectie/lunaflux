///|
fn stochastic_physical_program(
  plan : @luna_sampling_strategy.SamplingDevicePlan,
) -> @elementwise_physical_ir.StochasticProgram raise SamplingBoundaryError {
  guard @elementwise_physical_ir.StochasticProgram::new(
      values=plan.vocabulary_size(),
      owners=256,
      subgroup_lanes=32,
      maximum_top_k=plan.maximum_top_k(),
    )
    is Some(program) else {
    raise Invalid(DeviceSamplingPlan)
  }
  program
}

///|
fn stochastic_final_source(
  program : @elementwise_physical_ir.StochasticProgram,
) -> String {
  let source = StringBuilder()
  for step in program.final_steps() {
    source.write_string(
      match step {
        OrderedSoftmax =>
          "    const double maximum = (double)__bfloat162float(row_logits[candidate_tokens[0]]); double total = 0.0;\n" +
          "    for (int rank = 0; rank < retained_top_k; ++rank) { const double value = (double)__bfloat162float(row_logits[candidate_tokens[rank]]); const double weight = exp((value - maximum) / p.temperature); candidate_weights[rank] = weight; total += weight; }\n"
        RequireFiniteMass =>
          "    if (!(total > 0.0) || !isfinite(total)) { results[row * 2] = -1; results[row * 2 + 1] = -3; return; }\n"
        MinimalProbabilityPrefix =>
          "    int retained = retained_top_k; if (p.flags & 2) { const double target = p.top_p * total; double cumulative = 0.0; retained = 0; while (retained < retained_top_k && cumulative < target) { cumulative += candidate_weights[retained]; ++retained; } if (retained < 1) retained = 1; }\n"
        SumRetainedMass =>
          "    double retained_total = 0.0; for (int rank = 0; rank < retained; ++rank) retained_total += candidate_weights[rank];\n"
        CounterDraw =>
          "    const uint64_t random_bits = luna_counter_v2(p.seed, p.sample_index); const double unit = (double)(random_bits >> 11) * 1.1102230246251565e-16; const double threshold = unit * retained_total;\n"
        OrderedInverseCdf =>
          "    double cumulative = 0.0; int chosen = retained - 1; for (int rank = 0; rank < retained; ++rank) { cumulative += candidate_weights[rank]; if (threshold < cumulative) { chosen = rank; break; } }\n"
        PublishSample =>
          "    results[row * 2] = candidate_tokens[chosen]; results[row * 2 + 1] = -1;\n"
      },
    )
  }
  source.to_string()
}
