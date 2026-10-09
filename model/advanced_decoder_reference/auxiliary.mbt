///|
/// Expands the immutable auxiliary plan into its exact correctness-oracle
/// stage order. DSpark ordering mirrors main projection/noise embedding,
/// ordered decoder stages, base output head, Markov adjustment, then confidence.
pub fn auxiliary_stage_order(
  auxiliary : @advanced_decoder_execution_plan.AuxiliaryPredictionPlan,
  limits~ : ReferenceLimits,
) -> ReadOnlyArray[AuxiliaryReferenceStage] raise AdvancedDecoderReferenceError {
  let stages : Array[AuxiliaryReferenceStage] = []
  match auxiliary {
    MultiTokenPrediction(layer_count~) => {
      if layer_count <= 0 || layer_count > limits.max_iterations {
        raise InvalidAuxiliaryOrder
      }
      stages.push(MtpInputProjection)
      for stage in 0..<layer_count {
        stages.push(AuxiliaryDecoderStage(stage))
      }
      stages.push(AuxiliaryOutputHead)
    }
    MultiTokenPredictionWithDspark(layer_count~, dspark~) => {
      if layer_count <= 0 ||
        layer_count > limits.max_iterations ||
        dspark.target_layer_ids().is_empty() ||
        dspark.block_size() <= 0 {
        raise InvalidAuxiliaryOrder
      }
      stages.push(DsparkMainProjection)
      stages.push(DsparkNoiseEmbedding)
      for stage in 0..<layer_count {
        stages.push(AuxiliaryDecoderStage(stage))
      }
      stages.push(AuxiliaryOutputHead)
      stages.push(DsparkMarkovAdjustment)
      stages.push(DsparkConfidence)
    }
  }
  ReadOnlyArray::from_array(stages)
}
