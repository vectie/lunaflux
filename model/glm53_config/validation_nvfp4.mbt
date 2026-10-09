///|
/// The installed ModelOpt checkpoint quantizes routed experts only. This
/// descriptor describes storage, not BF16 activation semantics or a CUDA
/// schedule. Actual per-tensor payload/scale shapes remain owned by weight
/// binding; accepting this config cannot reclassify a BF16 tensor as FP4.
fn validate_flash_nvfp4(
  quant : Map[String, Json],
) -> Unit raise Glm53ConfigError {
  validate_known(quant, NumericFormat, name => {
    name
    is ("config_groups"
    | "quant_algo"
    | "kv_cache_scheme"
    | "producer"
    | "quant_method"
    | "ignore")
  })
  check_string(quant, "quant_method", "modelopt", NumericFormat)
  check_string(quant, "quant_algo", "NVFP4", NumericFormat)
  guard require_value(quant, "kv_cache_scheme", NumericFormat) is Null else {
    raise invalid(NumericFormat)
  }
  let groups = require_object(quant, "config_groups", NumericFormat)
  validate_known(groups, NumericFormat, name => name == "group_0")
  let group = require_object(groups, "group_0", NumericFormat)
  validate_known(group, NumericFormat, name => {
    name is ("targets" | "weights" | "input_activations" | "output_activations")
  })
  let targets = require_string_array(group, "targets", NumericFormat)
  guard targets is ["Linear"] else { raise invalid(NumericFormat) }
  for name in ["input_activations", "output_activations"] {
    guard require_value(group, name, NumericFormat) is Null else {
      raise invalid(NumericFormat)
    }
  }
  let weights = require_object(group, "weights", NumericFormat)
  validate_known(weights, NumericFormat, name => {
    name is ("num_bits" | "type" | "group_size" | "dynamic" | "symmetric")
  })
  check_int(weights, "num_bits", 4, NumericFormat)
  check_string(weights, "type", "float", NumericFormat)
  check_int(weights, "group_size", 16, NumericFormat)
  check_bool(weights, "dynamic", false, NumericFormat)
  check_bool(weights, "symmetric", true, NumericFormat)
  let producer = require_object(quant, "producer", InertMetadata)
  validate_known(producer, InertMetadata, name => name is ("name" | "version"))
  check_string(producer, "name", "modelopt", InertMetadata)
  guard require_string(producer, "version", InertMetadata) != "" else {
    raise invalid(InertMetadata)
  }
  let ignored = require_string_array(quant, "ignore", NumericFormat)
  guard !ignored.is_empty() else { raise invalid(NumericFormat) }
  let seen : Map[String, Unit] = Map([])
  for name in ignored {
    guard name != "" && !seen.contains(name) else {
      raise invalid(NumericFormat)
    }
    seen[name] = ()
  }
}
