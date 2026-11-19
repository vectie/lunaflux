# Chat over one prepared session

This transport adapter reuses LunaFlux's bounded HTTP parser, bearer policy,
OpenAI ingress, canonical handoff/events and JSON/SSE response renderer. It owns
one request credit and preallocated parsing/presentation storage. It imports no
model, CUDA or robot implementation. A caller supplies the prepared model's
generate and readiness callbacks.

The initial route is `POST /v1/chat/completions` with JSON `Content-Length`, a
configured bearer token. Omitted/false `stream` produces one JSON response;
`stream:true` delivers OpenAI SSE chunks as model events arrive. Chunked requests and pipelining are
not accepted by the existing bounded ingress. Concurrent arrivals get 429;
unsupported requests get 400, missing/wrong credentials get 401 and model/deadline
failures before response headers get 503. A buffered canonical Failed event
renders as500; streaming failure uses the canonical SSE error record. A connection
carries one request. The model adapter decides
which sampling and prompt profile it supports.

`Service::with_handler` separates the authenticated HTTP exchange from request
admission. A model adapter receives the live HTTP view, request ID, canonical
event owner, response-selection callback and emit callback. After admission it
calls `select_response(streaming)` once, before emitting events. It scopes its
own parser/handoff and cancels
native work before returning on cancellation. The transport still owns the
single request credit, timeout/disconnect watcher and canonical response
renderer. Selection is retained in the request exchange; it adds no duplicate
JSON parser or model state to transport. It imports no model or media code. `Service::new` remains the scoped
text-admission convenience constructor. Its body ceiling is 1 MiB; a custom
handler may select a larger ceiling within the existing HTTP parser's 16 MiB
maximum. No upload, JSON parsing or media work belongs to token scheduling.

```mbt check
///|
async fn _serve_scope(
  service : @session_chat.Service,
  listener : @socket.TcpServer,
) -> Unit {
  defer service.close()
  service.serve(listener)
}
```

The caller owns the listener, auth policy and backend session. Stop/cancel serve
and let its connection tasks retire before closing the service, policy or model.
The callback must retire native work on error/cancellation before it returns.
`is_ready` includes both the backend's readiness and the single request credit.
The service observes peer EOF/further input after a complete request, cancels
that request, then admits another only after callback cleanup. TCP half-close is
treated as disconnect in this initial single-request transport.

The timeout includes HTTP receipt, ingress, inference and response delivery.
Buffered success headers are written only for a completed response. Streaming
headers accompany the first nonempty rendered event; token chunks follow before
completion, with terminal usage and exactly one `[DONE]` on canonical completion
or failure. A timeout, callback failure or write failure after headers closes the
connection without appending a second HTTP response or false completion marker.
Before headers, the existing HTTP error mapping applies. Bounded renderer
copy quanta are obtained from its live view. The canonical ingress owns/releases
the HTTP view after it publishes the handoff; transport cleanup respects that
ownership transfer.

Network tests cover repeated buffered stop/length responses, canonical usage,
auth/invalid-request rejection, capacity, timeout, disconnect and reuse. The
streaming test holds completion until the client receives UTF-8 token content,
then checks terminal tails/usage, stop/length and canonical failure, post-header
timeout/disconnect cleanup and a later buffered reply. These tests use a
scripted callback and do not establish model inference or hardware timing.
Continuous batching, model discovery and health routes remain
outside this initial adapter.
