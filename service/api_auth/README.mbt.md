# Luna API authentication hook

`service/api_auth` owns a transport-neutral startup policy for one bounded
credential. Construction accepts a caller-owned `FixedArray[Byte]` range,
validates a configured capacity from 1 through 4096 bytes, and copies the
nonempty credential into exact-length private fixed storage. The capacity is
an ingress bound, not comparison padding. Mutating or retaining the
source afterward cannot change the policy.

The package deliberately has no disabled policy. An outer service that allows
unauthenticated operation represents that startup choice by the absence of a
`LunaApiAuthPolicy`; it cannot be confused with an empty credential, which is
rejected.

`verify` first authenticates the private policy invariant, then validates the
caller range with subtraction-safe arithmetic. Every valid range then performs
exactly the configured credential length's number of scalar byte comparisons. A wrong
length is folded into the accumulated mismatch; missing bytes compare as zero
and supplied bytes beyond the credential length are ignored. Every length and
mismatch position therefore has the same comparison count, and no mismatch
ends the loop early. The scalar decision exposes only accepted or rejected
predicates. The policy exposes no credential, storage, string,
digest, epoch, or configured-bound getter and has no `Debug` implementation.
Errors contain only typed rule and issue enums, never offsets or payload bytes.
`close` overwrites every cell in the complete fixed policy storage, invalidates
its scalar bounds, and is idempotent. Because the policy is reference-owned,
the wipe also invalidates every stale verifier alias without exposing state.

This transport-neutral package does not implement
HTTP Bearer parsing, a native preface, tenants, quota, authorization, sessions,
or a server. The reusable HTTP server owns Bearer parsing and consumes this
policy before semantic request admission.

For a reactor-owned header path, `LunaApiAuthVerifierWorkspace` is the
authoritative cooperative owner. `begin_verify` records only a declared
length; it never retains caller storage. The caller streams at most the first
credential-length bytes through `LunaApiAuthVerificationWrite::push_byte`.
Every successful push performs exactly one credential comparison and reports
one work unit. After `finish`, Work advances each missing zero-padding
comparison under its startup step budget, then charges one final decision
publication. Total work is therefore exactly the credential length plus one for
every declared length. Bytes beyond the credential length are deliberately not
accepted because length mismatch has already made the decision Rejected.

Write and Work authenticate a nonwrapping workspace generation. Busy owners,
incomplete or extra writes, stale aliases, epoch exhaustion, terminal take,
abort, and reuse are typed. Accepted and Rejected remain pinned until
`take_decision`; Failed remains pinned until `abort`. The synchronous `verify`
compatibility method drives the same scalar comparison primitive in one call
and must not be used as a reactor quantum.
Closing the verifier invalidates outstanding Write/Work epochs and delegates
the same policy wipe; later verification remains fail-closed.

Focused evidence is run with:

```sh
service/api_auth/validate-luna-api-auth.sh
```
