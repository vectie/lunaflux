///|
const APPROVED_SELECTED_LOGIT_TOLERANCE : Double = 0.01

///|
struct ApprovedCorpusCase {
  selected_ids : FixedArray[Int]
  selected_logits : FixedArray[Double]
  continuation : FixedArray[Int]
}

///|
fn approved_json_int(value : Json) -> Int raise ApprovedBf16PhysicalError {
  guard value is Number(number, repr=_) && number.to_int().to_double() == number else {
    raise InvalidInput("reference corpus integer is not exact")
  }
  number.to_int()
}

///|
fn approved_json_double(value : Json) -> Double raise ApprovedBf16PhysicalError {
  guard value is Number(number, repr=_) else {
    raise InvalidInput("reference corpus logit is not numeric")
  }
  number
}

///|
fn approved_corpus_case(corpus : Json) -> ApprovedCorpusCase raise {
  guard corpus is Object(root) &&
    root["model_cases"] is Array([Object(first), ..]) &&
    first["input_ids"] is Array([input]) &&
    approved_json_int(input) == 1 &&
    first["sampled_logits"] is Object(logits) &&
    first["greedy_continuation"] is Array(continuation_values) else {
    raise InvalidInput("reference corpus first model case is malformed")
  }
  let selected_ids : FixedArray[Int] = [
    0, 1, 2, 42, 99, 333, 897, 1031, 2049, 2185, 2999,
  ]
  let selected_logits = FixedArray::make(selected_ids.length(), 0.0)
  for index, token in selected_ids {
    guard logits.get(token.to_string()) is Some(value) else {
      raise InvalidInput("reference corpus selected logit is missing")
    }
    selected_logits[index] = approved_json_double(value)
  }
  let continuation = FixedArray::make(continuation_values.length(), 0)
  for index, value in continuation_values {
    continuation[index] = approved_json_int(value)
  }
  if continuation != [1031, 2185, 688, 2844] {
    raise InvalidInput("reference corpus continuation identity mismatch")
  }
  { selected_ids, selected_logits, continuation }
}

///|
fn compare_approved_selected_logits(
  expected : FixedArray[Double],
  actual : FixedArray[Double],
) -> Double raise ApprovedBf16PhysicalError {
  if actual.length() != expected.length() {
    raise NumericalMismatch("selected-logit count mismatch")
  }
  let mut maximum = 0.0
  for index, expected_value in expected {
    let error = (actual[index] - expected_value).abs()
    if error > APPROVED_SELECTED_LOGIT_TOLERANCE {
      raise NumericalMismatch("selected logit \{index} exceeded BF16 tolerance")
    }
    if error > maximum {
      maximum = error
    }
  }
  maximum
}
