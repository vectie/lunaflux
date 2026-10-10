# Single-contiguous-request decoder inputs

`SerialDecoderFrame` connects an existing validated scheduler wire frame to
fixed-capacity decoder input ports. It does not build a model, schedule requests,
or contain CUDA or family-specific branching.

Startup owns counts, token IDs, sequence offsets/slots, recurrent/sparse reset,
output count and selected-row payloads,
with an optional startup-selected absolute-position payload. Position values
are `committed_start + row`, including chunked prefill and decode; only the
live prefix is transferred. Legacy frames retain their eight-port ABI.
Each staged step requires one request,
exact context continuation, the loaded generation and in-vocabulary tokens.
Token-producing steps currently require greedy sampling. Unsupported batching,
prefix state and sampling are rejected rather than silently approximated.

Device completion commits history only after error ports and the sampled token
have been checked. Failure poisons history; cancellation aborts the pending
completion writer. Retiring a successful request explicitly resets the next
request's state. Wire retention uses a startup-owned slot, not boxed optional
value-type views in the token path.

This is worker plumbing, not proof of full-model inference or GPU arithmetic.

With `all_position_outputs=true`, startup reserves selected-row storage for
every input row. Token-producing stages select `0 .. live_rows` rather than
only the last row; ordinary generation remains unchanged. A canonical completion
cannot contain multiple samples and explicitly rejects that operation. A retired
verification coordinator consumes the result vector instead. `discard_stage`
aborts its writer without advancing committed request identity or position, but
only after the enclosing device-state restore has retired. Execution failures
must poison history, not be treated as a rejected draft.
