///|
fn matrix_affine_source(
  index : String,
  law : @precision.Float32AffineLaw,
) -> String {
  @arithmetic.scalar_expression(law.expression(), [
    "mixes[\{index}]",
    "scale[2]",
    "base[\{index}]",
  ]) catch {
    _ => abort("invalid closed matrix affine program")
  }
}

///|
fn matrix_sum_helper(
  law : @precision.SumReductionLaw,
  width : Int,
  symbol : String,
) -> String raise DecoderHyperConnectionCudaSourceError {
  match law {
    OrderedFloat32 => ""
    ButterflyFloat32 => {
      guard width > 1 && width <= 16 && (width & (width - 1)) == 0 else {
        raise InvalidNumericalContract
      }
      "__device__ __forceinline__ float \{symbol}(const float *values, int stride) { float partial[\{width}]; for (int i = 0; i < \{width}; ++i) partial[i] = values[i * stride]; for (int span = \{width / 2}; span > 0; span >>= 1) for (int i = 0; i < span; ++i) partial[i] = __fadd_rn(partial[i], partial[i + span]); return partial[0]; }\n"
    }
  }
}
