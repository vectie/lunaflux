///|
/// Checked startup envelope for semantic-plan admission.
///
/// The representation is opaque so callers cannot fabricate a value above the
/// package's hard ceilings. Limits are checked before numeric-schema mapping,
/// operation ownership, capability collection, or canonical hashing.
pub struct ModelPlanLimits {
  priv max_operations : Int
  priv max_numeric_tensors : Int
  priv max_total_value_inputs : Int
  priv max_value_inputs_per_operation : Int
  priv max_runtime_inputs_per_operation : Int
  priv max_outputs_per_operation : Int
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
let model_plan_hard_max_operations : Int = 4096

///|
let model_plan_hard_max_numeric_tensors : Int = 65536

///|
// Dense Llama's admitted 454-layer ceiling emits 9,086 value inputs. Keep the
// plan envelope independently bounded while covering that complete public
// model-spec range and leaving a small, explicit margin for later dense plans.
let model_plan_hard_max_total_value_inputs : Int = 16384

///|
let model_plan_hard_max_value_inputs_per_operation : Int = 64

///|
let model_plan_hard_max_runtime_inputs_per_operation : Int = 16

///|
let model_plan_hard_max_outputs_per_operation : Int = 1

///|
let model_plan_hard_max_canonical_bytes : Int = 16 * 1024 * 1024

///|
/// Constructs a checked plan envelope no larger than LunaFlux's hard startup
/// ceilings. The aggregate input ceiling must cover the per-operation ceiling.
pub fn ModelPlanLimits::new(
  max_operations~ : Int,
  max_numeric_tensors~ : Int,
  max_total_value_inputs~ : Int,
  max_value_inputs_per_operation~ : Int,
  max_runtime_inputs_per_operation~ : Int,
  max_outputs_per_operation~ : Int,
  max_canonical_bytes~ : Int,
) -> ModelPlanLimits raise PlanValidationError {
  if max_operations <= 0 || max_operations > model_plan_hard_max_operations {
    raise InvalidLimit(Operations)
  }
  if max_numeric_tensors <= 0 ||
    max_numeric_tensors > model_plan_hard_max_numeric_tensors {
    raise InvalidLimit(NumericTensors)
  }
  if max_total_value_inputs <= 0 ||
    max_total_value_inputs > model_plan_hard_max_total_value_inputs {
    raise InvalidLimit(TotalValueInputs)
  }
  if max_value_inputs_per_operation <= 0 ||
    max_value_inputs_per_operation >
    model_plan_hard_max_value_inputs_per_operation {
    raise InvalidLimit(ValueInputsPerOperation)
  }
  if max_runtime_inputs_per_operation <= 0 ||
    max_runtime_inputs_per_operation >
    model_plan_hard_max_runtime_inputs_per_operation {
    raise InvalidLimit(RuntimeInputsPerOperation)
  }
  if max_outputs_per_operation <= 0 ||
    max_outputs_per_operation > model_plan_hard_max_outputs_per_operation {
    raise InvalidLimit(OutputsPerOperation)
  }
  if max_canonical_bytes <= 0 ||
    max_canonical_bytes > model_plan_hard_max_canonical_bytes {
    raise InvalidLimit(CanonicalBytes)
  }
  if max_total_value_inputs < max_value_inputs_per_operation {
    raise InvalidLimit(IncoherentLimits)
  }
  {
    max_operations,
    max_numeric_tensors,
    max_total_value_inputs,
    max_value_inputs_per_operation,
    max_runtime_inputs_per_operation,
    max_outputs_per_operation,
    max_canonical_bytes,
  }
}

///|
/// Default bounded envelope used by production builders and ordinary tests.
pub fn ModelPlanLimits::production() -> ModelPlanLimits {
  {
    max_operations: model_plan_hard_max_operations,
    max_numeric_tensors: model_plan_hard_max_numeric_tensors,
    max_total_value_inputs: model_plan_hard_max_total_value_inputs,
    max_value_inputs_per_operation: model_plan_hard_max_value_inputs_per_operation,
    max_runtime_inputs_per_operation: model_plan_hard_max_runtime_inputs_per_operation,
    max_outputs_per_operation: model_plan_hard_max_outputs_per_operation,
    max_canonical_bytes: model_plan_hard_max_canonical_bytes,
  }
}

///|
fn ModelPlanLimits::check_operation_count(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_operations {
    raise LimitExceeded(Operations)
  }
}

///|
fn ModelPlanLimits::check_numeric_tensor_count(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_numeric_tensors {
    raise LimitExceeded(NumericTensors)
  }
}

///|
fn ModelPlanLimits::check_total_value_inputs(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_total_value_inputs {
    raise LimitExceeded(TotalValueInputs)
  }
}

///|
fn ModelPlanLimits::check_value_inputs_per_operation(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_value_inputs_per_operation {
    raise LimitExceeded(ValueInputsPerOperation)
  }
}

///|
fn ModelPlanLimits::check_runtime_inputs_per_operation(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_runtime_inputs_per_operation {
    raise LimitExceeded(RuntimeInputsPerOperation)
  }
}

///|
fn ModelPlanLimits::check_outputs_per_operation(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_outputs_per_operation {
    raise LimitExceeded(OutputsPerOperation)
  }
}

///|
fn ModelPlanLimits::check_canonical_bytes(
  self : ModelPlanLimits,
  count : Int,
) -> Unit raise PlanValidationError {
  if count > self.max_canonical_bytes {
    raise LimitExceeded(CanonicalBytes)
  }
}
