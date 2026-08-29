# Bounded HTTP/1 request framing

`service/http1` is the transport-neutral HTTP/1 request-framing and
authentication foundation. It is deliberately not a listener, socket loop,
JSON parser, request translator, or multi-connection server. The separate
OpenAI compatibility and online TCP packages compose JSON translation,
response bytes, and serialized listener ownership above it.

The accepted protocol is intentionally small and exact:

- HTTP/1.1 `POST` to `/v1/responses` or `/v1/chat/completions`;
- one `Content-Length` whose decimal value fits the configured body bound;
- one `Content-Type` with the exact media type `application/json`;
- one `Authorization` value using the `Bearer` scheme and an ASCII token68
  credential; and
- an optional set of other syntactically valid ASCII headers.

Header names and the media type are ASCII case-insensitive. The parser permits
at most one leading ASCII space after a colon and rejects trailing whitespace,
obs-fold, controls, NUL, non-ASCII head bytes, malformed names, duplicate
authority headers, duplicate or conflicting lengths, every
`Transfer-Encoding`, unsupported methods/routes/versions/media types, and
configured head/body overflow. JSON body semantics are not inspected here.

## Operational request owner

A disjoint bounded owner recognizes only bodyless HTTP/1.1 `GET /healthz` and
`GET /readyz`. It accepts no transfer encoding and permits only an absent
`Content-Length` or one exact zero length. Unsupported methods, paths, versions,
trailing bytes, malformed headers, and configured head overflow fail to a
fixed 400 response. This owner does not widen or share state with the inference
parser.

The corresponding response primitive selects only fixed payload-safe JSON
responses for healthy, unhealthy, ready, not-ready, and bad-request outcomes.
It exposes length, status, and bounded copying, never caller-selected content.
Listener ownership and live lifecycle projection remain in `service/online_tcp`.

## Ownership and work

`LunaHttp1Workspace` allocates exactly `max_head_bytes + max_body_bytes` byte
cells and two capacity-one authentication capability slots at startup. The
body begins at the fixed `max_head_bytes` offset, so a forged or variable head
length cannot substitute body layout. The caller's offered `FixedArray[Byte]`
is never retained.

Credential verification immediately overwrites the presented credential bytes.
Operation reset invalidates the prior lease and scalar lengths without an
additional O(head+body) sweep; later input overwrites reusable storage.
Terminal authentication close overwrites the complete fixed request storage,
invalidates outstanding work, and delegates the shared expected-policy wipe.
OpenAI server and pool owners invoke that close only after their
listener/service drain reaches terminal closure.

Head receipt stops on the byte that completes `CRLFCRLF`. Even when the same
offered range also contains body or a pipelined request, `offer` returns the
exact head consumption and leaves the tail caller-owned. The stored head is
then scanned cooperatively. Only after all structural checks and the immutable
`LunaApiAuthPolicy`'s cooperative verifier accept the credential does the work
enter `LunaHttp1ReceivingBody`; no body byte can be copied earlier. Exact body
completion likewise leaves a pipelined tail unconsumed. A later positive offer
to an already-ready Work is a typed trailing-byte failure.

One accepted head byte, one scanned or validated head byte, one authentication
comparison/publication step, one accepted body byte, or one scalar phase
transition consumes one HTTP work unit. The authentication child is fixed to a
one-unit budget. `offer` and `progress` never exceed the configured outer
budget; `last_work_units` and `total_work_units` report the exact accounting.
The frozen two-byte responses fixture consumes 347 total units for budgets 1,
17, and 65536.

`finish_input` records transport EOF. An incomplete head fails immediately;
an EOF recorded while head validation or authentication is pending becomes a
body premature-end failure if the authenticated length still requires bytes.
Source range errors authenticate the Work first and are transactional: they
consume no byte and do not poison the request. Protocol failures enter an
authenticated Failed phase, replay a payload-safe rule/issue value, and remain
abortable. Abort is the only recovery to Idle. A Ready Work transfers its epoch
to one immutable `LunaHttp1View`; release invalidates the View and enables
workspace reuse. Every View payload accessor authenticates staleness before
checking bounds.

The View exposes only `route`, `body_length`, indexed `body_byte_at`, `is_live`,
and `release`. It has no credential, head, policy, epoch, array, `Bytes`, or
`String` projection. The credential is supplied to authentication one scalar
byte at a time and is never materialized as a public object.

Validation precedence is frozen as Work epoch, transactional source range,
head envelope and ASCII line structure, method, route, version, encountered
header syntax/duplicates/transfer encoding, missing Authorization then
Content-Length then Content-Type, credential verification, and finally body
EOF/trailing checks.

Focused evidence is run with:

```sh
service/http1/validate-luna-http1-allocations.sh
```

## Fixed response primitive

The same package also owns a disjoint preallocated response primitive. It does
not accept caller payloads. `begin_event_stream` builds exactly the 96-byte
HTTP/1.1 200 head for `text/event-stream`, `Cache-Control: no-cache`, and
`Connection: close`; the OpenAI online Server composes semantic SSE event bytes
only after this fixed head is ready. `begin_error` selects only one finite
payload-free JSON response:

- authentication: 401 with a fixed `WWW-Authenticate: Bearer` challenge and a
  41-byte `authentication_error` body;
- validation: 400 with a 37-byte `validation_error` body;
- capacity: 429 with a 35-byte `capacity_error` body; or
- service: 503 with a 34-byte `service_error` body.

Every error has an exact fixed `Content-Length`, `application/json`, and
`Connection: close`. There is no API for an error message, field, identifier,
credential, request byte, or vendor string, so input payload cannot be
reflected into an error response.

`LunaHttp1ResponseWorkspace` owns exactly 167 byte cells, the largest supported
response. Its Work writes one fixed byte per unit plus one publication unit;
the resulting immutable View exposes only status, length, indexed byte access,
bounded copy into caller-owned fixed storage, liveness, and release. Copying is
limited by the response workspace's startup step budget. Invalid copy ranges
are transactional, and stale authentication precedes all range checks. The
response owner allocates only at startup and is reusable after abort or View
release.
