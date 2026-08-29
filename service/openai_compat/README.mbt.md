# OpenAI compatibility

This package is LunaFlux's transport-neutral outbound compatibility codec. It
cooperatively maps an authenticated `@luna_event.LunaEventView` to canonical
Server-Sent Events for either the Responses or Chat Completions wire shape.
It does not own transport acknowledgement authority: retiring or acknowledging
the source event remains the caller's responsibility.

`LunaOpenAICompatWorkspace` is startup-preallocated. Its constructor copies a
bounded UTF-8 deployment model alias and a bounded safe-ASCII response-ID
prefix into private fixed storage. `required_byte_cells` and
`required_reference_cells` expose checked aggregate sizing before allocation.
The response sequence must be nonzero; a server must drain before exhausting
its `UInt64` sequence space.

`begin` authenticates one semantic event and returns an opaque, generation-
bound `LunaOpenAICompatWork`. A workspace remains pinned until Work aborts or
its resulting View is released. Failed work remains authenticated and requires
abort. Ready construction performs a final semantic authentication and then
detaches the event reference. The completed framed bytes therefore remain
readable after the semantic owner retires, but carry no ACK authority.

The encoder writes canonical bytes directly into its fixed backing. It uses
compile-time fixed literal tables plus scalar model, ID, numeric, and event
payload reads; it constructs no dynamic String, Bytes, JSON AST, or intermediate
event object. JSON strings preserve validated non-ASCII UTF-8, escape quote and
backslash, use the short escapes for backspace, tab, newline, form feed, and
carriage return, and encode other controls as lowercase `\\u00xx`.

Each `progress` call consumes at most its `LunaOpenAICompatStepBudget`. One
charged unit is exactly one of:

- one authenticated semantic setup scalar;
- one part or phase transition;
- one literal, model, response-ID, numeric, or JSON-escape output byte;
- one source payload-byte load and classification; or
- one decimal-divisor preparation step.

`last_work_units` and `total_work_units` report these exact units. The same
event produces identical bytes and total work for budgets 1, 17, and 65536.
`LunaOpenAICompatView::copy_chunk_to` rejects zero-length copies and limits each
copy to the workspace step budget, preventing a caller from spinning without
transport progress or monopolizing the reactor with an unbounded final copy.

Responses emits `response.created`, `response.output_text.delta`,
`response.usage`, `response.completed`, and `response.failed`. Chat Completions
emits assistant-role, content, usage, finish, and error chunks. The terminal
`data: [DONE]\n\n` record appears only after Completed or Failed semantic
events, exactly once.

## Inbound request compatibility

`LunaOpenAIInboundWorkspace` is the transport-neutral inbound aggregate. It
retains one authenticated `@http1.LunaHttp1View`, cooperatively parses a strict
bounded JSON subset, renders Chat messages through an explicit startup-owned
`LunaOpenAIChatTemplate`, and writes the prompt and stops directly into an
authoritative typed `LunaTextRequestHandoffStorage`. The resulting
`LunaOpenAIInboundView` exposes a generation-bound handoff lease; it does not
serialize or reparse a canonical Luna request frame. It never constructs
`GenerateRequest`, `TextInput`, a JSON AST, `String`, or `Bytes` on the
production request path. Its sizing methods include the prompt, model alias,
typed semantic child, fixed stop-token and stop-string tables, and the full
configured stop-string byte envelope before construction.

Responses requires `model`, a string `input`, and `stream: true`. Chat
Completions requires `model`, `stream: true`, and a nonempty `messages` array
whose objects contain exactly one `role` and one nonempty string `content`.
Both routes additionally accept bounded `temperature`, `top_k`, `top_p`,
`seed`, string-or-string-array `stop`, and integer-array `stop_token_ids`.
Absent optional values preserve the validated startup sampling and seed.
Explicit values are scanned one byte per charged transition into scalar or
fixed-capacity storage. JSON syntax and numeric representation are checked
while parsing; typed sampling construction and the semantic handoff validate
each decoded range, UTF-8 stop, duplicate stop, and cache field once before
publication.

Object key order and JSON whitespace are unrestricted; unknown and duplicate
keys, trailing commas, non-text content, malformed or overflowing numbers, and
unsupported fields fail closed. Chat roles allow an optional initial system message followed by strict
user/assistant alternation, and the final client message must be user. JSON
strings validate raw UTF-8 and decode escapes, including paired UTF-16
surrogates, one charged input or emitted byte at a time.

The template copies seven bounded UTF-8 segments for system, user, and
assistant framing plus the assistant cue. Its exact rendered-prompt ceiling is
checked before every Chat output byte. Responses bypasses the template and is
bounded by `InferenceLimits.max_text_bytes`.

Inbound Work reauthenticates the HTTP view before every charged transition,
including every typed semantic-validation quantum and the final publication
step. Only then does it release the HTTP view. Failure deterministically
releases every retained child authority; abort returns the workspace to idle.
Ready transfers one opaque typed lease whose release returns all workspace
storage for exact-generation reuse. Each `progress` call is bounded by
`LunaOpenAICompatStepBudget`; nested semantic progress uses a fixed budget of
one.

The configured `DeadlineBudget` is the original relative wire budget. This
package never samples or rebases a receipt timestamp. A reusable HTTP server
must begin an authenticated request-admission receipt before accepting byte 1,
poll it while HTTP/auth/JSON/template/semantic validation advances, and submit
the typed handoff with that same receipt. Request admission samples that
receipt once, preserves its exact absolute deadline, and transfers the prompt
and validated semantic lease without copying them. The inbound constructor deliberately requires
`max_new_tokens < context_ceiling`: a text request with equality has zero input
capacity and is rejected before any HTTP authority is retained.

Inbound parsing and typed admission do not own semantic-delivery
acknowledgement. Native framed ingress remains the canonical framed wire route
and is unchanged by this typed OpenAI handoff.
Likewise, the outbound caller may invoke `SemanticEvent.delivered` only after
the entire compatibility View has been confirmed by the transport; a partial
SSE write is not delivery. Reusable HTTP server integration and inbound request
receipt wiring remain separate packages/slices.
