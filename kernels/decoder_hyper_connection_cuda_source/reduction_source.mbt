///|
fn control_square_source(
  law : @precision.SquareReductionLaw,
) -> (String, String, String) raise DecoderHyperConnectionCudaSourceError {
  match law {
    OrderedFloat32 =>
      (
        "float square_sum = 0.0f;", "square_sum = __fadd_rn(square_sum, __fmul_rn(x, x));",
        "",
      )
    CompensatedFloat32 => {
      let (init, update, finish) = @reduction.dot_accumulator_source(
        CompensatedFloat32,
        accumulator="square_sum",
      ) catch {
        _ => raise InvalidNumericalContract
      }
      (init, "const float product = __fmul_rn(x, x); " + update, finish)
    }
  }
}

///|
fn control_dot_source(
  law : @precision.DotReductionLaw,
  operand : String,
  width : Int,
) -> (String, String, String) raise DecoderHyperConnectionCudaSourceError {
  match law {
    OrderedFloat32 =>
      (
        "float dot = 0.0f;",
        "dot = __fadd_rn(dot, __fmul_rn(\{operand}, weight));",
        "",
      )
    CompensatedFloat32 => {
      let (init, update, finish) = @reduction.dot_accumulator_source(
        CompensatedFloat32,
        accumulator="dot",
      ) catch {
        _ => raise InvalidNumericalContract
      }
      (
        init,
        "const float product = __fmul_rn(\{operand}, weight); " + update,
        finish,
      )
    }
    StripedTreeFloat32(partitions~) => {
      guard partitions >= 2 &&
        partitions <= 256 &&
        partitions <= width &&
        (partitions & (partitions - 1)) == 0 &&
        width % partitions == 0 else {
        raise InvalidNumericalContract
      }
      (
        "float dot = 0.0f; float dot_partials[\{partitions}] = {};",
        "dot_partials[column % \{partitions}] = __fadd_rn(dot_partials[column % \{partitions}], __fmul_rn(\{operand}, weight));",
        "for (int span = \{partitions / 2}; span > 0; span >>= 1) for (int p = 0; p < span; ++p) dot_partials[p] = __fadd_rn(dot_partials[p], dot_partials[p + span]); dot = dot_partials[0];",
      )
    }
  }
}
