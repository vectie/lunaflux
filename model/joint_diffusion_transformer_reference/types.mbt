///|
pub(all) enum JointTinyConditioningSource {
  TextConditioning
  KeyframeConditioning
  ReferenceConditioning
} derive(Debug, Eq)

///|
pub(all) suberror JointTinyTransformerReferenceError {
  InvalidLimit
  InvalidPlan
  InvalidRequest
  InvalidConditioningContract
  InvalidConditioningShape(JointTinyConditioningSource)
  InvalidDenseShape
  InvalidMlpShape
  InvalidWeightShape
  InvalidStepIndex(Int)
  ElementBudgetExceeded
  NonFiniteInput
  NonFiniteComputation
} derive(Debug, Eq)

///|
pub struct JointTinyTransformerLimits {
  priv maximum_rows : Int
  priv maximum_hidden_width : Int
  priv maximum_feed_forward_width : Int
  priv maximum_elements : Int
}

///|
pub struct JointTinyConditioningMatrix {
  priv source : JointTinyConditioningSource
  priv rows : Int
  priv width : Int
  priv values : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
pub struct JointTinyConditioningInput {
  priv text : JointTinyConditioningMatrix
  priv keyframe : JointTinyConditioningMatrix
  priv reference : JointTinyConditioningMatrix
} derive(Debug, Eq)

///|
pub struct JointTinyDense {
  priv input_width : Int
  priv output_width : Int
  priv weights : ReadOnlyArray[Double]
  priv bias : ReadOnlyArray[Double]
} derive(Debug, Eq)

///|
pub struct JointTinyMlp {
  priv up : JointTinyDense
  priv down : JointTinyDense
} derive(Debug, Eq)

///|
pub struct JointTinyTransformerWeights {
  priv plan_digest : @spec.PlanDigest
  priv text_projection : JointTinyDense
  priv keyframe_projection : JointTinyDense
  priv reference_projection : JointTinyDense
  priv video_input_projection : JointTinyDense
  priv audio_input_projection : JointTinyDense
  priv query_projection : JointTinyDense
  priv key_projection : JointTinyDense
  priv value_projection : JointTinyDense
  priv attention_output_projection : JointTinyDense
  priv video_mlp : JointTinyMlp
  priv audio_mlp : JointTinyMlp
  priv video_output_projection : JointTinyDense
  priv audio_output_projection : JointTinyDense
}

///|
pub struct JointTinyDenoiseStepResult {
  priv plan_digest : @spec.PlanDigest
  priv step_index : Int
  priv initial_video : ReadOnlyArray[Double]
  priv initial_audio : ReadOnlyArray[Double]
  priv packed_projected : ReadOnlyArray[Double]
  priv attended : ReadOnlyArray[Double]
  priv video_prediction : ReadOnlyArray[Double]
  priv audio_prediction : ReadOnlyArray[Double]
  priv updated_video : ReadOnlyArray[Double]
  priv updated_audio : ReadOnlyArray[Double]
  priv video_sigma : Double
  priv video_next_sigma : Double
  priv audio_sigma : Double
  priv audio_next_sigma : Double
}
