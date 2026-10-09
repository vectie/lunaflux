///|
pub(all) enum DecoderHyperConnectionCudaOperation {
  BlockControlV1
  StagedFunctionAndCollapseV1
  StagedSinkhornV1
  PreReduceV1
  PostCombineV1
  HeadReduceV1
} derive(Debug, Eq)

///|
pub(all) enum DecoderHyperConnectionNumericalContract {
  OrderedF32BlockControlV1
  OrderedF32FunctionBf16CollapseV1
  OrderedF32PositiveSinkhornV1
  OrderedF32PreReduceBf16RoundV1
  OrderedF32PostCombineBf16RoundV1
  OrderedF32HeadControlReductionBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) suberror DecoderHyperConnectionCudaSourceError {
  InvalidStreamWidth
  InvalidHiddenWidth
  InvalidMaximumRows
  InvalidIterationCount
  InvalidEpsilon
  InvalidFunctionSymbol
} derive(Debug, Eq)

///|
/// Family-neutral deterministic mHC block-control CUDA source. This is source
/// evidence only and grants no compiler, artifact, loader, or launch authority.
pub struct DecoderHyperConnectionCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv hidden_width : Int
  priv maximum_rows : Int
  priv sinkhorn_iterations : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral staged mHC function source. It produces post controls, the
/// positive row-softmax matrix that precedes Sinkhorn, and one BF16 collapsed
/// branch input. It grants no execution or artifact authority.
pub struct DecoderHyperConnectionFunctionCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv hidden_width : Int
  priv maximum_rows : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral staged positive-matrix Sinkhorn source. Its input is already
/// row-softmaxed and epsilon-shifted by the function stage.
pub struct DecoderHyperConnectionSinkhornCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv maximum_rows : Int
  priv sinkhorn_iterations : Int
  priv control_epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral deterministic mHC pre-reduction CUDA source. It grants no
/// compiler, artifact, loader, or launch authority.
pub struct DecoderHyperConnectionPreReduceCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv hidden_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral deterministic mHC post-combination CUDA source. It grants
/// no compiler, artifact, loader, or launch authority.
pub struct DecoderHyperConnectionPostCombineCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv hidden_width : Int
  priv maximum_rows : Int
  priv function_symbol : String
  priv source : Bytes
}

///|
/// Family-neutral deterministic mHC head-reduction CUDA source. It grants no
/// compiler, artifact, loader, or launch authority.
pub struct DecoderHyperConnectionHeadReduceCudaSource {
  priv operation : DecoderHyperConnectionCudaOperation
  priv numerical_contract : DecoderHyperConnectionNumericalContract
  priv stream_width : Int
  priv hidden_width : Int
  priv maximum_rows : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv function_symbol : String
  priv source : Bytes
}
