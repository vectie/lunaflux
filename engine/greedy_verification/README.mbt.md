# Exact greedy draft verification

This package is a pure, model-neutral prefix decision. Input row zero is the
already-emitted seed; target output row `i` predicts the token after input row
`i`. Every accepted draft input after the seed must equal the preceding target
output. On the first mismatch the target correction becomes the next seed,
not another committed KV row. All-accepted verification likewise emits a bonus
token without pretending its KV already exists.

Stop tokens and remaining output length truncate the accepted input prefix,
even when additional drafts matched. The caller must restore all persistent
state and replay only that prefix when `rollback_required()` is true. Publication
waits for both ranks' device and host-frontier transactions. This package never
performs those effects or claims that a draft is independently authoritative.

The decision is a value type, owns no arrays, and has a zero-allocation warmed
native regression. It validates the whole retired vector, including later
invalid samples after an early output limit. It implements greedy target
verification only; stochastic rejection sampling needs proposal probabilities
and correction distributions and is not provided here.

```mbt check
///|
test {
  let result = @greedy_verification.verify_greedy_prefix(
    [10, 11, 12, 13],
    [11, 12, 7, 15],
    remaining_new_tokens=10,
    vocabulary_size=16,
    stop_tokens=[],
  )
  assert_eq(result.accepted_input_rows(), 3)
  assert_eq(result.emitted_token_count(), 3)
  assert_eq(result.next_seed(), 7)
  assert_true(result.rollback_required())
}
```
