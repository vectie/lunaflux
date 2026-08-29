# NVIDIA validation campaign — 2026-08-27

## 2026-08-28 final30 exact-current-source qualification

The metadata-clean portable source archive SHA-256 is
`1b951694414dbd9fc9d796eb9532580d82e0bb6a101ad289194ec58cd8d12eaa`.
On the Linux NVIDIA host it passes the warning-denied 506-task native check and
2,297/2,297 tests, including platform-specific tests skipped on macOS. The only
compiler warning is the known vendored MoonBit async C declaration warning for
`posix_spawn_file_actions_addchdir_np`; first-party MoonBit warnings remain
denied.

The sealed current-source physical evidence is
`/tmp/lunaflux-final30-current-source-physical-20260828`. `RESULT.txt` records
`outcome=passed`, `exit_status=0`, and `terminal_stage=complete`. Source-file,
worker, baseline launch, prefix launch, and evidence-manifest SHA-256 values are
`a29842dacf52802063cd29353f5697d2aa38ae12c1d4fd2e11e3440819f2c92a`,
`e4aa8ae78538c166fdf65ea256151f1d2b8c6b4b0bc990948c8558fd7fed6c25`,
`50f7b0d25635700ee480bc1a1ee012fa1f0dc54f39285659e27edb20aba40edb`,
`df1ea2725b51a1c03ec17694e8cf126cab817b42e90c4878fcfd691a4fd9fffa`,
and
`94852d96c82390ffa427d2ce630c89086bec4f64b8395c431ed31d2fccd9a8cc`.
Every `FILES.sha256` entry rehashes successfully. CUDA 13.1.115 remained
offline-only compiler authority and the request path used no JIT.

The campaign passes spawned BF16 execution, the exact listener admission
transition, concurrent broad serving, backpressure, cancellation, foreign and
malformed isolation, same-owner recovery, restart, non-routable OpenAI
qualification, a measured qualification-only benchmark trial, and eight-token
prefix reuse. All runtime stderr files are empty, network and KV resources are
balanced, both listeners and children close, GPU memory returns to 15/22 MiB,
and no compute process remains. The separate restored-host Phase 7 diagnostic
again rejects the heterogeneous `sm120`/`sm75`, no-peer, no-NCCL topology before
resource authority. The exact-final29 24-hour soak was active with empty
stdout/stderr throughout this campaign and later passed as recorded below.

This is positive exact-current-source single-GPU BF16 evidence only. It does
not claim positive I8/FP8, homogeneous tensor-parallel/NCCL, public TLS, the
81-trial vLLM/SGLang comparison, approved OCI/SBOM/provenance, or release
promotion.

## 2026-08-28 final29 exact-current-source qualification

The metadata-clean portable source archive SHA-256 is
`28d00c97a5b21b14d0951aab728254880330a11e2d42d3889d9eb6c750791a90`.
On the Linux NVIDIA host it passes the warning-denied 503-task native check,
2,291/2,291 tests, the pinned worker-executable and child-control sanitizer
gates, activation-allocation audit, process ABI boundary, and worker authority
boundary. The only compiler warning is the known vendored MoonBit async C
declaration warning for `posix_spawn_file_actions_addchdir_np`; first-party
MoonBit warnings remain denied.

The sealed current-source physical evidence is
`/tmp/lunaflux-final29-current-source-physical-20260828`. `RESULT.txt` records
`outcome=passed`, `exit_status=0`, and `terminal_stage=complete`; source-file,
worker, baseline launch, prefix launch, and evidence-manifest SHA-256 values are
`96c196e18e39383a9962ba91cb66a633abba925c4eb8686ff90f17aae127e9e1`,
`cd6560ce9a4f638b452d8ca1cd084e999cc845d89f414a21974e7129cf262547`,
`8315adc7876da13d251b3b356e627cf2894f416db4776a7ac9bda11b89035a75`,
`a5dfafd208e64aa1c61223a84f38ffd7983f6461821d5e6e33072e319c1da235`,
and
`62da6e7221312d3d9bc6898e2af274c53ef89e5f4a3d71dae2056f648b8373b9`.
Every `FILES.sha256` entry rehashes successfully. CUDA 13.1.115 remained
offline-only compiler authority and the request path used no JIT.

Spawned BF16 execution produced tokens `1031,2185` and closed all authority.
The native listener now proves `Listening/Ready -> Connected/NotReady ->
Listening/Ready` with healthy service state, ordered accepted/token/token/usage/
completed events, one balanced accept/disconnect, all 32 KV pages free, and
closed listener/child authority. Broad qualification passes concurrent work,
active-plus-waiting backpressure, cancellation, foreign and malformed
rejection, same-owner recovery, restart, five balanced accepts/disconnects,
two rejections, four completions, one cancellation, and exact cleanup.
Non-routable OpenAI qualification and eight-token physical prefix reuse also
pass. GPU memory returned to 15/22 MiB and no compute process remained.

The final29 Phase 2/3 v3 fast and timer diagnostics each pass 2,300 waves and
4,600 requests with 4,465 completions, 135 cancellations, 16 rejections, 41
balanced accepts/disconnects, positive batching/backpressure/cancellation
counters, `kv_free8`, and closed child authority. The final29 24-hour soak also
passes. `RESULT.txt` records `outcome=passed`, `exit_status=0`, and
`elapsed_millis=86400012`; the worker's single terminal line measures
86,400,002 milliseconds across 43,200 cycles and 86,400 requests, with every
required wave counter positive, 833 balanced accepts/disconnects, and final
resources `queue0,active0,kv_used0,kv_free8,pending0,child_closed`. Runtime
stderr is empty. Every entry in `FILES.sha256` rehashes successfully and the
manifest SHA-256 is
`a49609d819e143adf1e63d99f0ee97014a58d095baaffd0f9a3c2eb43ff8788f`.
The immutable remote archive is
`/home/jiaanguo/lunaflux-final29-phase23-soak-24h-20260828-pass.tar.gz`;
the independently downloaded local copy is
`/private/tmp/lunaflux-final29-phase23-soak-24h-20260828-pass.tar.gz`.
Both are 2,064,860 bytes with SHA-256
`3a105bb7ef9665c86e8dabec1808ef61da838c3579bcb4929e05948e623d4fb0`.
The prior final7 soak is preserved as infrastructure-interrupted after
host loss removed its transient unit before `RESULT.txt`.

This closes the exact-current-source bounded BF16 physical campaign only. It
does not claim positive I8/FP8, homogeneous tensor-parallel/NCCL, public TLS,
the 81-trial vLLM/SGLang comparison, OCI/SBOM/provenance, or LunaNexa release
integration.

## 2026-08-28 final7 full Linux and NVIDIA qualification

The exact portable source archive has SHA-256
`bd1c9a275435ab25a0fb11ce539fde1745945bf542e6389250d4383a50c6e313`.
It passes `moon info`, `moon fmt --check`, warning-denied native check, and
2,065/2,065 native tests on the Linux NVIDIA host. The only test-build stderr
is the known declaration warning in the vendored MoonBit async C runtime for
`posix_spawn_file_actions_addchdir_np`; first-party MoonBit warnings remain
denied. The native-gate evidence archive is retained locally at
`/private/tmp/lunaflux-native-gate-evidence-20260828-final7-bd1c9a275435-rerun1-pass.tar.gz`
with SHA-256
`02605679f83e8ee8b76c0aa98adffd2312a7048fb89919f73234435d1e303375`.

The complete current-source campaign rebuilt worker SHA-256
`381e4a393922502cae9de67f0f0205033661d0c8a3ae1e8db6bb1c0c7517e859`
and materialized baseline and prefix launch SHA-256 values
`54fe9873a9c81f2fc6e36cbc215dfe45bdd649b24fc335996130ec86e6a3c610`
and
`5515fdc793c3ebb544dab51f2d3de89d491e3e68631d7d2e4c8e58c115f2bea9`.
It compiled and authenticated the 21-operation AOT set with exact CUDA
13.1.115, retained offline-only compiler authority, and used no request-path
JIT. Spawned execution, the native listener, broad bounded BF16 serving,
caller-authorized non-routable Responses, a one-request timestamp
qualification, and eight-token physical prefix reuse all passed. Broad serving
covered two concurrent requests, eight backpressure observations,
cancellation, foreign and malformed rejection, same-owner recovery, restart,
drain, 22 events, five balanced accepts/disconnects, two rejections, restored
all 32 KV pages, and deterministic listener/child closure.

The sealed result is `outcome=passed`, `exit_status=0`,
`terminal_stage=complete`. Its source inventory and evidence manifest SHA-256
values are
`58f963b0dee2af0ca478115900ed320eb42e1da80613627af87010a37ec8ba61`
and
`56599642163a25771c6d3b58c2a2f5023ac3c0ebf52b9cc1f75979a88f8f2ae6`.
Every manifest entry was rehashed after download. The local archive is
`/private/tmp/lunaflux-physical-evidence-20260828-final7-bd1c9a275435-rerun1-pass.tar.gz`
with SHA-256
`2d1441f36929ee8bdfeeac7bc5ffdc11e9fdb06de7ee6b7c9ed350cf30ad09c6`.
The benchmark measured 41 ms queue, 63 ms first token, 73 ms service, 51 ms
decode, and 114 ms end to end. This is a qualification measurement, not the
required counterbalanced LunaFlux/vLLM/SGLang baseline comparison. Responses
evidence remains `non_routable`, `not_ready`, loopback plaintext, without TLS
or a health endpoint.

An independent supported-hardware sweep passed 128 CUDA primitive cycles, all
eight BF16 kernel families, paged attention with maximum absolute error
`0.0104694`, a 128-cycle capture-required 12-kernel graph, the five-case,
160-launch shape matrix, and two byte-identical approved-model runs. Those
model runs authenticated weights and artifacts, produced greedy tokens
`1031,2185,688,2844`, and closed resources. The sweep manifest SHA-256 is
`72f6a42ce7d0ae3d8f9c5c70eb8a8091a38c0026ea681cb90540fa29477e1229`;
the local archive SHA-256 is
`8bb86a6cce09fc299bc5c6461b09cd3626cac850e420015fda2ad991afa27cb0`.
GPU 0 returned exactly to 15 MiB and no compute process remained.

The final7 Phase 7 topology diagnostic is retained locally with archive
SHA-256
`855e253214b5d9ed727811dac66a3548f541a2a261d03fac30f37428f6fcdd90`.
It reports the exact heterogeneous `sm120`/`sm75` inventory, no peer access in
either direction, and missing `libnccl.so.2`; admission rejects before context,
allocation, communicator, or rank authority. The final7 Phase 8 archive
SHA-256 is
`d31bed457d6fd8a08f417def68581b3ea443e31911b4eaedf729d3b424386bb8`.
It rejects unsupported `sm120` I8 before context or device allocation and
preserves exact GPU/process balance. These are correct fail-closed results,
not positive I8 or tensor-parallel/NCCL validation.

The digest-pinned Phase 2/3 v3 24-hour soak was launched under
`lunaflux-phase23-soak-v3-final7-20260828.service`. Its stdout/stderr remain
empty and no result exists. Host loss removed its transient unit, so it is
preserved infrastructure-interrupted evidence, not a pass.

## 2026-08-28 final6 broad current-source qualification

The exact portable source archive has SHA-256
`5e60bf5d532ac5ccccfe9a88a67d0b5b55533cc6af13388c7ff485665a486031`
and is retained on the NVIDIA host at
`/home/jiaanguo/lunaflux-current-source-20260828-final6.tar.gz`. Its fresh
source and sealed evidence directories are
`/home/jiaanguo/lunaflux-physical-source-20260828-final6-5e60bf5d532a` and
`/home/jiaanguo/lunaflux-physical-evidence-20260828-final6-5e60bf5d532a`.
The locally reverified evidence archive is
`/private/tmp/lunaflux-physical-evidence-20260828-final6-5e60bf5d532a-pass.tar.gz`
with SHA-256
`4e5cb30e672399f2f155f54d0a19172be255f3d545f2daac42891ada2ad1696e`.
This section is post-campaign documentation and is not part of the byte-exact
tested source.

The local phase boundary passed `moon info`, `moon fmt`, warning-denied native
check, and 2,065/2,065 native tests. The Phase 2/3 v3 real-worker smoke passed
three waves and six requests; its fast diagnostic passed 2,300 waves and 4,600
requests with 4,465 completions, 135 cancellations, 16 typed rejections, 41
accepts/disconnects, balanced queue/active/KV/pending resources, and a closed
child. The canonical v3 policy digest is
`ab268f305c71658b53b9f8347eb77a79d1fd4c414fc01faa0492a48e3886751a`.
The two historical v2 24-hour passes remain valid for their frozen source, but
the v3 malformed-rejection policy requires a fresh 24-hour pass before it is
current-source soak promotion evidence.

Linux execution subsequently exposed a soak-harness-only portability defect:
the pipeline fixture still opened macOS `/private/tmp`. The runner now requires
an explicit approved fixture root. Portable final7 source archive
`bd1c9a275435ab25a0fb11ce539fde1745945bf542e6389250d4383a50c6e313`
passes 2,065/2,065 local tests and a cold Linux real-worker v3 smoke. Its
24-hour run is active as
`lunaflux-phase23-soak-v3-final7-20260828.service`; evidence is
`/home/jiaanguo/lunaflux-phase23-soak-v3-evidence-20260828-final7-bd1c9a275435`.
The frozen worker-service and worker-echo SHA-256 values are
`f90ff634f4229b5e76835419e9f42848924fa94c46124808f089092e3abff5a3`
and
`46551a195e42b5063366cf9aef91da68ed8eea352c3a4f40bf8f00223ab3d10f`.
Runtime stdout/stderr were empty at launch. This is a running diagnostic, not
terminal promotion evidence.

Two preceding wrapper attempts are preserved but not promoted. The first was
stopped before timed execution because it mixed normal Moon build status into
the runtime-stderr channel. The second separated the channels but the frozen
final6 harness aborted before listener creation when it opened `/private/tmp`
on Linux. That failure directly motivated the explicit-root final7 correction.

On the RTX 5060 Ti (`sm120`), CUDA 13.1.115 compiled and authenticated the
21-operation offline AOT set. The campaign rebuilt child SHA-256
`c742fcdeb6b124f55bfe0fa748ea895a8dc0a8bd3ee11276ef989e8399065f52`,
then materialized baseline and prefix launches
`2911984a644b8163d5cf06a74724702718a1021aa3068591bf471fc2f31b62e6`
and
`49a8e479e0fde9b02ad6ff741bff8996b6b50e7cf0f91b908a6141c967193d24`.
The source inventory remained stable, compiler authority remained offline-only,
and the request path used no JIT. The sealed result is `outcome=passed`,
`exit_status=0`, `terminal_stage=complete`; its source inventory and evidence
manifest SHA-256 values are
`88a680c2994541f851961940705153978244a83e76728b995bdcdda214ad76a6`
and
`5cefe085bce81262ae9df6e0be635421161597bc065ff3736ca7048f762bc60f`.
Every manifest entry was independently reverified after download.

The broad physical BF16 qualification exercised two concurrent requests,
active-plus-waiting saturation with eight backpressure observations,
cancellation isolation, typed foreign-model rejection, malformed-frame
connection isolation, same-owner recovery after malformed input, fresh-owner
restart, and drain. It completed four of five admitted requests with one
cancellation, emitted 22 canonical events, balanced five accepts with five
disconnects and two rejections, restored all 32 KV pages, and closed both
listener/child generations. The broad record SHA-256 is
`3f46837e0c71e059a7c52291a22121124f0e9f289859a9d337336c4461917a0a`.

The caller-authorized loopback Responses qualification again rejected missing
and wrong bearer credentials with 401, returned 200 for the correct credential,
emitted the five canonical Responses events, drained, refused post-drain
admission, and restored all resources. Its record SHA-256 is
`e9baa415eabe202328909d1ab6eee10530b0c20cbb3a71a3ddc84c86b9049505`;
it explicitly records `qualification_provenance=non_routable` and
`production_readiness=not_ready`. The physical prefix campaign reused eight
tokens and produced independently fixed outputs `1355,1240`; its record
SHA-256 is
`e51b09a15a124b9f81f8235e5b2dbfb65f385708620dd2ff4e1434bae18f1109`.
The one-request benchmark measured queue 41 ms, first token 62 ms, service 74
ms, decode 53 ms, and end-to-end 115 ms. It is a qualification measurement,
not the full counterbalanced LunaFlux/vLLM/SGLang comparison.

GPU framebuffer use returned exactly to its initial 15 MiB on GPU 0 and 16 MiB
on GPU 1, with no compute process. A preceding final5 archive failed only at a
stale static spelling guard after the Linux warning-denied listener cleanup;
no CUDA stage ran. Its sealed failure archive is retained locally at
`/private/tmp/lunaflux-physical-evidence-20260828-final5-7747f8311105-failure.tar.gz`
with SHA-256
`e0efeb9091d9ab228ab431371f0a75ac8f1859042961883b9d8a3b7fde9d0543`.
It is not promotion evidence.

The exact final6 Phase-7 diagnostic also exited zero with a typed rejection:
the two devices are heterogeneous `sm120`/`sm75`, peer access is unavailable
in both directions, and the NCCL library is missing. It opened no context,
allocation, communicator, or rank process. The sealed local archive is
`/private/tmp/lunaflux-phase7-evidence-20260828-final6-5e60bf5d532a.tar.gz`
with SHA-256
`daaf0b7dbfda343f9c380b9e784bde6fd0ad45b2bacf35c0b7cbb4d9a582a16b`.
GPU memory again remained exactly 15/16 MiB with no compute process.

The exact final6 Phase-8 I8 admission probe likewise produced the required
typed rejection on device 0: production I8 v1 accepts only exact `sm89` or
`sm90`, while the observed target is `sm120`. It rejected before reading a
module or opening a context/allocation, emitted empty stderr, and preserved
exact GPU memory/process balance. The sealed local archive is
`/private/tmp/lunaflux-i8-rejection-evidence-20260828-final6-5e60bf5d532a.tar.gz`
with SHA-256
`c0057606efbb3311fae9e1b5cf8bdbb0dcc9a92bf524e1844ac42ae1a8ad42ef`.
This closes the unsupported-hardware gate, not positive I8 numerics.

This campaign establishes current-source single-GPU BF16 execution and broad
bounded serving qualification. It does not establish public routability, TLS,
production health approval, the full pinned baseline comparison, final
OCI/SBOM/provenance/signing, current-v3 24-hour soak, positive I8, or positive
homogeneous tensor-parallel/NCCL evidence.

## 2026-08-28 authenticated serving, benchmark, and prefix qualification

The final portable source archive has SHA-256
`847c8493a4d43faf8517969be8780eace00380b69fa9c6b7894d3fd9e45b36ac`.
It is retained at
`/home/jiaanguo/lunaflux-current-20260828-openai-prefix-r2.tar.gz`; the two
fresh extracted source trees and evidence are beneath
`/home/jiaanguo/lunaflux-validation-20260828-openai-prefix-r2-847c8493a4d4`.
The locally reverified pass-evidence archive is
`/private/tmp/lunaflux-evidence-20260828-openai-prefix-r2-pass.tar.gz` with
SHA-256
`88cf855d31d04d9277755da548565c3e7eefcafbadfecfbb25716b3b4a840483`.
Its physical and full-suite manifest SHA-256 values are
`79835c8ab1c177a3c0b930367a61f96baa8e5aae50f6ec039648ae1366082b0e`
and
`0ff43d00fef9a618435d2f7e833a601156872fa1133b11de1f6a5ce06d4671d7`;
all 209 physical entries and all seven full-suite entries were independently
verified after download. This result section is post-campaign documentation and
is not itself part of the tested archive; the executable source is byte-exact.

The exact archive passed the warning-denied Linux native check and 2,038 of
2,038 native tests. The fresh physical tree rebuilt the canonical child from
that source, compiled the authenticated BF16 AOT set with CUDA 13.1.115, and
recorded source-file inventory
`7679e42b3e7f01d24067d5a717ce2a9fd1b3567d5b0272411a8c78e95265cc04`.
The child, baseline launch, and prefix launch SHA-256 values are respectively
`da511747509a8032aecfbed1e188a47e2a8f08970890dd2ed2eea5f2a267942b`,
`a61767b1038817ccb7ac67413ed49917ec0dc3660b647fe8914a6d817876d930`,
and
`5923e1be4d75c649839fb91459ef317dc4cb834650d3cf46ddf142bc4ca658c0`.
The source inventory remained byte-identical after materialization and campaign
build, and the request path used no JIT.

The spawned native listener again produced `1031,2185` with exact request,
network, KV, listener, child, and owner balance. The caller-authorized loopback
OpenAI Responses qualification rejected missing and wrong bearer credentials
with 401, returned 200 for the correct credential, emitted
`response.created`, `response.output_text.delta`, `response.usage`,
`response.completed`, and `[DONE]` in order, then drained and refused new
admission. It recorded three accepts/disconnects, two authentication
rejections, zero used KV pages, all 32 pages free, complete cleanup, explicit
`qualification_provenance=non_routable`, and
`production_readiness=not_ready`. Its record SHA-256 is
`bce96233d8009f7d7d88f7c0df4f38576798c0ea5df2537952c3d8b933072f34`.
TLS and a health endpoint were not tested or claimed.

The physical benchmark measured one pinned one-input/two-output request at the
real listener boundaries: queue 37 ms, first token 54 ms, service 57 ms,
decode 40 ms, and end-to-end 94 ms. Its record SHA-256 is
`bd8f2786d4d6e7fe5f04091b5c75f7f76bc8dccaaeea14aac21b97a2f3762396`.
This is a single LunaFlux qualification measurement, not the nine-profile,
three-trial, counterbalanced LunaFlux/vLLM/SGLang comparison. The authenticated
prefix launch then observed a cold first request and an eight-token hit on the
second request: two lookups, one hit, one miss, one publication, eight reused
tokens, independent expected outputs `1355,1240`, and complete cleanup. Its
record SHA-256 is
`d78565f4bbcf0225e7371ce4bc4c9673897bf02579ab80ad2742f99fac958664`.

An earlier source archive,
`913370731874a53154573b7cd9f2134dfc851ec3741a82ae910b559b69e7ad48`,
failed closed twice at the benchmark listener bind after the OpenAI 401 paths
left server-side port 8080 sockets in `TIME_WAIT`. Those results were not
promoted. LunaFlux now centralizes all five TCP listener constructors with
`SO_REUSEADDR` while never enabling `SO_REUSEPORT`; an independent live probe
confirmed that a concurrent exact-address listener still receives
`EADDRINUSE`, while immediate same-address restart succeeds. The rejected
evidence is retained locally at
`/private/tmp/lunaflux-evidence-20260828-openai-prefix-r1-failure.tar.gz`
with SHA-256
`e25e7905d6b3ab67a7d5fbad19f07081428ec076094e76c90dfb8e5ffe1d54ac`.

The final GPU inventory returned exactly to 15 MiB on the RTX 5060 Ti and
16 MiB on the RTX 2080, with no compute process. Runtime, compiler, verifier,
serving, OpenAI, benchmark, and prefix stderr are empty. The native compiler
still reports the known vendored `moonbitlang/async` C declaration warning for
`posix_spawn_file_actions_addchdir_np`; LunaFlux MoonBit warnings remain
denied. Positive I8, homogeneous tensor-parallel/NCCL, OCI/SBOM/signing, TLS,
production health/readiness/drain approval, and external vLLM/SGLang comparison
remain separate open gates.

## 2026-08-28 exact-current-source qualification

The final portable source archive has SHA-256
`99887e5f4687889fd30f3927508a4adc49ff0b1f052117f87bca9b8c069d9e83`.
It is retained on the NVIDIA host at
`/home/jiaanguo/lunaflux-current-source-99887e5f4687.tar.gz`; its extracted
campaign root is
`/home/jiaanguo/lunaflux-validation-20260828-current-99887e5f4687`.
The sealed 222-file evidence archive has SHA-256
`ce3521079a238f738b172438175205ba626d0db3ef095e33b43c84935ce69105`
and is retained both on that host and locally at
`/private/tmp/lunaflux-current-99887e5f-evidence.tar.gz`. The evidence manifest
and result record have SHA-256 values
`27b4067f506835669b897b983a3b270d1e4ef3cd2f733bd4002b0801fd7962cd`
and
`14097db6cf0279c16455160572e5e2653d012a10f01f2c81a948fd4907a9c128`.

On Linux, the exact source passed the warning-denied native check (434 tasks)
and 2,010/2,010 native tests. Ten sanitizer/native-ABI gates passed, including
the CUDA ordered executor, CUDA peer boundary, NCCL ABI, process maintenance,
approved filesystem, TCP alias, and monotonic-clock checks. The Linux compiler
emitted only the known vendored async-runtime C declaration warning for
`posix_spawn_file_actions_addchdir_np`; MoonBit warnings remained denied.
Three schema/materialization gates and three tensor-parallel static gates could
not run their positive controls with Ubuntu's ripgrep 13 build because it lacks
PCRE2. The exact scripts passed locally with PCRE2; this is recorded as a host
tooling limitation, not silently counted as a remote pass.

GPU 0, an RTX 5060 Ti with exact `sm120`, passed the 128-cycle CUDA primitive,
all eight generated BF16 families, paged attention, the capture-required
12-kernel graph, and the five-case shape matrix (160 launches). Two approved
model runs were byte-identical with stdout SHA-256
`92b1247b137345b74043a4e7ad30f9ce4001c8e5a2c6953c32044c8e7b7732ff`:
eleven selected logits stayed within maximum absolute error `0.0005459413`,
the greedy tokens were `1031,2185,688,2844`, same-page KV persistence passed,
graph capture was required, and resources closed. The current-source parent
also passed spawned execution and native listener serving against the
reauthenticated preserved launch and child whose SHA-256 values are
`96567a4a432993fe5e7e968fabc7a4b15e1b23de1fa89ebae181639dfdea92d6`
and
`4a1e58ba9fc99b4e53e476e5417c4f17ea8c6807fccc2cfda16ea3721cfe7776`.
The child was not rebuilt from this archive, so the evidence makes no such
claim. Concurrent native-framed slow/fast-client progress, the 10,000-request
Phase 2/3 balance campaign, and the soak diagnostic also passed.

The campaign ended with no LunaFlux or GPU compute process. GPU framebuffer
use returned exactly to the initial 15 MiB on GPU 0 and 16 MiB on GPU 1.
Positive I8 remains untested because production I8 v1 accepts exact `sm89` or
`sm90`, while this host is `sm120`; its probe rejected before resource
authority as required. Positive tensor parallel/NCCL remains untested because
the two GPUs are heterogeneous (`sm120`/`sm75`), peer access is unavailable in
both directions, and the NCCL runtime is absent; the product diagnostic
rejected before context, allocation, communicator, or rank-process creation.
Physical prefix reuse remains open because the preserved authenticated launch
has prefix reuse disabled and an eight-token input ceiling. Performance remains
open because the new canonical benchmark collector has no physical
protocol/process adapter yet. The prior two 24-hour soak passes remain valid;
this exact-source campaign ran the finite 10,000-request balance proof rather
than another 24-hour soak.

The host exposes `nvidia-ctk` but no Docker, Podman, Buildah, Syft, Cosign, or
Trivy executable. It therefore cannot produce or approve the final OCI image,
SBOM, provenance signature, or root-filesystem scan from this campaign. That
deployment-tooling result is independent of the successful native CUDA path.

## 2026-08-28 current-source requalification addendum

The portable current-source archive with macOS extended attributes excluded
has SHA-256
`974d50938b1fbc31e6a2c37aaaadf5f034d9e93dd48db428618cc3003c77de5a`.
It is retained with the immutable r20 campaign at
`/home/jiaanguo/lunaflux-validation-20260828-physical-numeric-r20-portable`.
The campaign passed all 12 bounded cases on GPU 0, an RTX 5060 Ti with exact
`sm120`: a 431-task warning-denied native check, 128-cycle CUDA primitive,
eight BF16 kernel families, paged attention, a 128-cycle capture-required
12-kernel graph, the five-case/160-launch shape matrix, and two byte-identical
runs of the approved tiny BF16 model. The model runs authenticated weights and
AOT artifacts, matched eleven selected logits with maximum absolute error
`0.0005459413`, produced greedy tokens `1031,2185,688,2844`, preserved
same-page KV state, required graph capture, and closed resources. No compute
process remained and framebuffer use returned to the 15 MiB baseline. The
result SHA-256 is
`e155ca692e99c50f4bf855f38886657af5eb2fd520160fdefb91036b7f7b1a55`;
the local evidence-only archive SHA-256 is
`e0775796c3a095a223e0592e05375f364a3c1f8fa93c228c7ec0ab9aeb37207e`.

A separate fresh listener requalification retained at
`/home/jiaanguo/lunaflux-validation-20260828-serving-requal-019ff4ef/evidence/serving-r1`
again passed the exact `Accepted`, `Token(1031)`, `Token(2185)`, `Usage`,
`Completed` sequence, one accept/disconnect, restored 32-page KV balance,
empty stderr, and complete listener/child cleanup. Its immutable manifest
SHA-256 is
`64632084d872fb211b9f8aca67e7258117dd29b7be8e76abaab16b4fffef20fb`.
The approved physical campaign is explicitly cache-disabled and single-request,
so this is not cached-prefix execution evidence.

The fresh capability campaign is retained at
`/home/jiaanguo/lunaflux-validation-20260828-capability-gates-r1`; its local
evidence archive SHA-256 is
`aaca5f609ba0ca57115812d57918557db036beb53f0da6744233dbcdb9a5d27a`.
The product topology diagnostic rejected the heterogeneous sm120/sm75 pair
before creating a context, allocation, communicator, or rank process; peer
access is unavailable in both directions and `libnccl.so.2` is absent. The I8
probe built its exact sm89 CUBIN and the sm120 runtime rejected it before
resource creation under the exact sm89/sm90 policy. These are fail-closed
passes, not positive I8, tensor-parallel, or NCCL validation. Focused hardware-
policy tests passed 176/176 and no GPU process remained.

Two preceding source-transfer attempts, r18 and r19, are retained as packaging
failures: macOS provenance metadata was reconstructed as AppleDouble files on
Linux and MoonBit refused the invalid UTF-8 sidecars before execution. They are
not product failures or promotion evidence. R20 adds a pre-test AppleDouble
guard and uses the portable archive above.

## 2026-08-28 native-listener readiness addendum

The final listener-validation source archive is retained at
`/home/jiaanguo/lunaflux-validation-20260828-integrated-r16-serving/evidence/serving-r3/source-r17-final.tar.gz`.
Its SHA-256 is
`08b0139de9d5c2b9e70da27e3f8de2cc4fc2f91571458bdab889b4b824f65a9b`.
The Linux warning-denied targeted check completed 117 tasks and the targeted
runtime/campaign suite passed 34/34 tests before execution. The campaign reused
the unchanged r14 launch and canonical child digests recorded below; it did not
recompile or rematerialize the approved model or kernel set.

The r3 campaign bound the real production native TCP pipeline listener on
loopback, observed health separately from readiness, submitted the exact pinned
one-token prompt, and received `Accepted`, `Token(1031)`, `Token(2185)`,
`Usage`, and `Completed(TokenLimit)` in order. It recorded one admission and
completion, one network accept and disconnect, zero failures/rejections/
cancellations/deadlines, zero queued or active requests, zero used KV pages,
all 32 pages free, closed listener and child ownership, empty stderr, and no
remaining GPU compute process. The canonical stdout SHA-256 is
`4a045cd5a9c95cb81b94339c12213993a6bca819a1b92038321b63476984b8c3`;
its self-bound serving record is
`093d4ce83c108af62e2dca9c02ba03af394fce65551f5d164f588d5fb2ce3bb1`.

The preserved r1 and r2 attempts failed closed at the event-read diagnostic
with empty stderr and complete cleanup. They exposed a validator-only
cooperative scheduling bug: the client blocked for bytes while the same task
still owned server-reactor progress. R3 interleaves bounded client reads with
owner progress. The combined immutable r1-r3 evidence archive has SHA-256
`c236f1626fc958fe51f1a24eeaff635a1f3bce367999cca2bf330d6f5228da8b`.

This proves one bounded native loopback traffic-readiness slice for the pinned
tiny BF16 request. It does not prove TLS, public-network exposure, concurrent
clients, broad shapes/contexts, cached-prefix execution, leak/soak behavior,
latency, throughput, or production release readiness.

## 2026-08-28 spawned-worker completion addendum

The final physical runtime source is retained at:

```text
/home/jiaanguo/lunaflux-validation-20260828-integrated-r14-clean/source
```

Its archive SHA-256 is
`385a958e41e86d367d3565002db674ee2177e47528f8714039cd08c7d9480eef`.
A cold Linux native check completed 431 tasks with warnings denied. The local
phase boundary passed 1,961/1,961 native tests. A code-identical snapshot with
only documentation and source-guard corrections is used for the final host
suite because the old static guard still searched for the former inline KV
expression after the physical binary run.

The final target-specific launch is retained beneath
`evidence/spawned-physical-r1/work/launch`. Its launch SHA-256 is
`96567a4a432993fe5e7e968fabc7a4b15e1b23de1fa89ebae181639dfdea92d6`,
and its canonical child SHA-256 is
`4a1e58ba9fc99b4e53e476e5417c4f17ea8c6807fccc2cfda16ea3721cfe7776`.
The offline binder authenticated the unchanged 21-operation sm120 candidate
set and nine-CUBIN compiled set, derived bootstrap
`2ddf387eb07a660f571a6552fe03a4407816dbf79b68b735520e234c5e291f33`,
and semantically preflighted the descriptor, policy, model, tokenizer,
manifest, modules, and child without invoking a compiler or opening CUDA.

The production one-argument runtime path then spawned that child on the RTX
5060 Ti and executed one request through the owned framed-service boundary.
The campaign produced exact tokens `1031,2185`, consecutive plan sequences
`1,2` for one-row/one-token prefill and same-page decode, the admitted
256-token context and positive emergency reserve, complete stream/service/
device/child cleanup, empty stderr, and no remaining GPU compute process. The
result SHA-256 is
`0f75a2399979210a4bcf7a641ed2ad32c1906746838dea8c14e56ba39d84cfba`.
Scheduler telemetry was accepted only in one of two exact states at semantic
event consumption: one live page, or the already-retired request with the
original 32-page balance restored.

This is physical spawned-worker token correctness for one pinned tiny BF16
model before listener publication. `traffic_readiness=0`, selected-logit
correctness is not observed by this campaign, and it does not promote TLS,
general serving, performance, positive I8, tensor parallelism, or a production
release. Positive I8 remains unavailable on sm120; the heterogeneous sm120/
sm75 pair still has no peer access or NCCL runtime.

## 2026-08-28 integrated-snapshot addendum

The final integrated source archive is retained at:

```text
/home/jiaanguo/lunaflux-validation-20260828-integrated-r9/source
```

Its archive SHA-256 is
`cfef45b89703e1dfe936b0cbee7eaa77ef4b1614172a9eac188a3c7115be60bb`.
On the Linux/CUDA host, `moon check --target native --deny-warn` exited zero
and `moon test --target native --deny-warn` passed 1,957/1,957 tests. The test
compiler emitted only the retained vendored
`moonbitlang/async` C warning for
`posix_spawn_file_actions_addchdir_np`; LunaFlux MoonBit warnings remained
denied. The OCI packaging, deterministic deployment-bundle assembly, atomic
release materialization, deployment-boundary, and spawned-execution static
contract gates all exited zero with empty stderr.

The same r9 source repeated the approved tiny-model physical campaign on the
sm120 RTX 5060 Ti. CUDA 13.1.115 compiled the authenticated 21-operation AOT
set; required graph capture executed the authenticated weights and artifacts;
eleven selected logits matched with maximum absolute error
`0.0005459412999999982`; greedy tokens were `1031,2185,688,2844`; same-page
KV persistence passed; and resources closed. The final device-process inventory
was empty. The root-bound concurrent OpenAI pool also passed its slow-client,
fast-client, connection-reuse, retirement, and drain campaign with empty
runtime stderr.

This r9 addendum predates the r14 spawned-worker completion recorded above.
Positive I8 is still unavailable on sm120, and the heterogeneous sm120/sm75
node has neither peer access nor `libnccl.so.2`; those paths continue to fail
closed.

## Decision

The private CUDA boundary, a complete synthetic captured paged-BF16 graph, a
bounded five-case shape matrix, and the pinned upstream tiny BF16 model passed
physical validation on GPU 0. The approved-model campaign compiled the exact
21-operation plan with CUDA 13.1.115, authenticated the weights and AOT
artifacts, matched eleven selected logits with maximum absolute error
`0.0005459413`, produced greedy tokens `1031,2185,688,2844`, preserved
same-page KV state, required graph capture, and closed all resources. This does
**not** prove broad/concurrent serving, throughput/latency, positive I8
execution, or tensor parallelism. The later r17 addendum proves only one
bounded native-listener request. The I8 path instead correctly rejects this `sm120` host
before opening execution resources, and the heterogeneous second GPU cannot
satisfy the tensor-parallel contract.

The clean approved-model source and evidence are retained at:

```text
/home/jiaanguo/lunaflux-validation-20260828-d3d2a556/source-d3d2a556651eefaf
/home/jiaanguo/lunaflux-validation-20260828-d3d2a556/evidence-20260827T161805Z
```

The source archive SHA-256 is
`d3d2a556651eefaf872e72a703a4afbb9bf54145c3f8ae9e698009fa5137064e`.
It contains 2,356 repository entries and no AppleDouble sidecars. Runtime,
compiler, and verifier stderr are empty; the runtime stdout SHA-256 is
`92b1247b137345b74043a4e7ad30f9ce4001c8e5a2c6953c32044c8e7b7732ff`.
The same Linux work tree passed the final warning-free native check and
1,936/1,936 native tests. Its full-validation manifest SHA-256 is
`aaf252382261dffa7893606492e264942a7180a97cec183e1847c507ede87509`.

The final bounded hardware sweep is retained at:

```text
/home/jiaanguo/lunaflux-validation-20260828-d3d2a556/hardware-sweep-20260827T162939Z
```

Its five-case BF16 shape matrix passed 40 capture-required launches with zero
error and closed resources. The I8 probe compiled its exact sm89 CUBIN but the
runtime correctly rejected GPU 0 as sm120 before artifact/resource admission.
The CUDA peer, device-topology, NCCL ABI, and NCCL lifecycle sanitizer gates
passed; physical inventory reports P2P read/write `NS` in both directions
between the sm120 RTX 5060 Ti and sm75 RTX 2080. `libnccl.so.2` and nccl-tests
are unavailable, so no physical NCCL collective is claimed. The 100-file
read-only sweep manifest SHA-256 is
`71a095d507ce4a04f1d2f9ff75ef4e6d1a3ca3b638eea751d9715604372ee806`;
GPU inventory was unchanged and the final compute-process file is empty.

The earlier `/home/jiaanguo/lunaflux-validation` tree was externally removed
on 2026-08-28 after the historical digests below had been recorded. Those
paths are historical identifiers, not currently retained remote directories.

The remote campaign evidence is retained at:

```text
/home/jiaanguo/lunaflux-validation/f8cbfe0df49f62028d2cad57dcdef5be0930ebadf0ca9493b7216176fcc46de3
```

The final post-integration source snapshot and Linux gate evidence are retained
at:

```text
/home/jiaanguo/lunaflux-validation/final-source-a94cafd03093de0b5d5890c54bd36309b329d23cd60d61eae2e296f91758a3bd
```

The subsequent residual-add AOT campaign is retained separately at:

```text
/home/jiaanguo/lunaflux-validation/aot-residual-20260827
```

The final paged-attention numerical rerun is retained at:

```text
/home/jiaanguo/lunaflux-validation/paged-attention-20260827-rerun3
```

The final current BF16-family fixture campaign is retained at:

```text
/home/jiaanguo/lunaflux-validation/bf16-families-final-20260827
```

The clean CUDA graph-capture rerun is retained at:

```text
/home/jiaanguo/lunaflux-validation/cuda-graph-82e3549dfa570c93db033d6bfb007fbcdbed63adb26702bd2c382cb7ae7656e7-final
```

The complete synthetic paged-BF16 graph campaign is retained at:

```text
/home/jiaanguo/lunaflux-validation/paged-bf16-graph-256205c68d5a59f4eb6d68661100a784d4a4e39f4661e288e9721f9724710cd3
```

The final integrated Linux/GPU validation snapshot is retained at:

```text
/home/jiaanguo/lunaflux-validation/final-current-993e9fd0d12015d4636a1f9d86a3f32911c03b5ca4d5adecf60c275b33bd131e
```

The final tokenizer/release/I8/portable-balance integrated snapshot superseding
that campaign is retained at:

```text
/home/jiaanguo/lunaflux-validation/final-current-8351d804617ab9a5
```

Its network-independent source archive SHA-256 is
`8351d804617ab9a5ca542d51b76685eefa2820394417e7a9f8852c9b9db41ae8`.
The archive includes the pinned Moon package cache, contains no AppleDouble
sidecars, and was extracted without network dependency resolution.

Its source archive SHA-256 is the digest in the directory name. A preceding
`3d407c...` attempt is preserved as packaging evidence but is invalid: macOS
AppleDouble sidecars reached Linux and were rejected as non-UTF-8 MoonBit
source before compilation. The corrected archive contains no sidecars.

The complete graph campaign's first evidence directory stopped before build
because non-interactive SSH did not place MoonBit on `PATH`. That infrastructure
failure is preserved. The unchanged source overlay passed in `evidence-rerun1`
after supplying the exact installed MoonBit path.

The first campaign attempt was preserved as a packaging failure: AppleDouble
entries introduced invalid UTF-8 before compilation. The later `c83f6d...`
clean pass was preserved but superseded by the final `82e354...` source set,
which also contains the graph operation-gate race probe. Neither earlier
campaign is the reported final evidence.

Earlier `paged-attention-20260827` and `-rerun1` attempts stopped at CUDA 13
probe-compilation compatibility, while `-rerun2` passed numerically but
retained a compiler warning on stderr. Their evidence was preserved; only
`-rerun3` is the clean pass reported below.

## Host inventory

- Ubuntu 22.04, NVIDIA driver 590.48.01, CUDA Driver API version 13010.
- GPU 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0, 16,311 MiB,
  BF16 supported.
- GPU 1: NVIDIA GeForce RTX 2080, compute capability 7.5, 8,192 MiB,
  BF16 unsupported.
- CUDA peer access is unavailable in both directions. The heterogeneous pair
  is therefore an unsupported tensor-parallel topology and must fail startup.

## Passed evidence

- `moon check --target native --warn-list -73-92 --deny-warn`: 351 tasks,
  exit 0. The two disabled diagnostics are compatibility warnings introduced
  by the host's newer MoonBit compiler; every other warning remains denied.
- `moon test --target native --warn-list -73-92 --deny-warn`: 1,835/1,835
  tests passed, exit 0.
- The final integrated `993e9fd0...131e` snapshot passed 405-task native check
  and 1,887/1,887 native tests with the same host-compiler compatibility mask.
  Its BF16 release/root assembly, symmetric-I8 numeric, dense-builder, inert
  admission, and production-execution software gates, CUDA ABI gate, both
  warmed device-worker allocation gates, ordered-executor ASan/UBSan gate,
  Phase 2/3 diagnostic boundary, and release-evidence boundary all exited 0.
  The allocation harness intercepts both the older libc allocator path and the
  newer Linux compiler's `moonbit_malloc_raw` path, with independent positive
  controls, so record allocations cannot evade the warmed-path measurement.
- The superseding `8351d804...41ae8` snapshot passed a 407-task Linux native
  check and 1,902/1,902 native tests under the same host-compiler compatibility
  mask. CUDA ABI, both device-worker allocation gates, ordered-executor
  ASan/UBSan, Phase 2/3 diagnostic, release assembly/evidence, tokenizer
  equivalence, numeric/I8 foundation, dense-I8 builder, inert I8 admission,
  production I8 execution, and I8 physical-probe boundary gates passed. Its
  corrected canonical temporary-root fixture also passed the complete
  10,000-request Phase 2/3 balance run on Linux.
- That same snapshot repeated the complete paged-BF16 graph for 128
  capture-required cycles with token `7`, maximum absolute error `0`, closed
  resources, and empty stderr. An exact `sm89` CUBIN was then supplied to the
  physical I8 probe on GPU 0; it exited nonzero with
  `production I8 v1 requires exact sm89 or sm90 device 0` before opening
  resources. This is positive fail-closed hardware-policy evidence, not I8
  numerical execution evidence.
- The final post-integration source archive (`a94cafd0...a3bd`) clean-built on
  the Linux host and passed 1,882/1,882 native tests. The same snapshot passed
  strict CUDA ABI, ordered-executor ASan/UBSan, CUDA graph lifecycle,
  kernel-bundle, I8 execution, release-bundle, and OCI gates with shell failure
  propagation enabled. Remote compiler stderr contains only the known vendored
  async-runtime `posix_spawn_file_actions_addchdir_np` declaration warning.
- CUDA Driver/cuBLASLt probe: 128 complete open/execute/close cycles, exit 0.
  Every cycle checked a 4 KiB host/device round trip, exact 2x3x2 BF16 GEMM
  output, events, PTX module/function loading, a direct AOT add-one launch, the
  ordered executor, completion polling, and reverse-order resource closure.
- The offline Luna CUDA AOT builder produced the exact sm120 BF16 residual-add
  specialization twice with byte-identical CUBIN output. The generated CUBIN
  then passed 128 physical cycles through the real ordered executor using its
  fixed three-pointer ABI and completion-event wait. Every cycle checked 16
  exact BF16 sums and closed all retained resources. This validates one kernel
  specialization, not a full graph.
- The digest-pinned reference paged-attention source passed a physical sm120
  differential probe for one mixed batch containing prefill and decode rows.
  It exercised the exact ten-pointer ABI, stable FP32 softmax, current-chunk
  reads, and fused split K/V cache writes. Maximum absolute BF16 error against
  the independent CPU referee was `0.0104694`; stderr was empty. This is one
  small correctness fixture, not leak, shape-matrix, or performance evidence.
- The current digest-pinned embedding, RMSNorm, positioned-RoPE, residual,
  QKV, dense-output, gated-MLP, and LM-head sources all compiled for sm120 and
  matched an independent BF16 referee across three adversarial live tokens.
  Fixture hashes were checked before compilation, the probe reported all eight
  families, and stderr was empty. This is reference-kernel numerical evidence,
  not full-model or performance promotion.
- The capture-required ordered-executor path passed 128 physical cycles on GPU
  0. Each cycle instantiated the exact startup-prepared add-one graph, selected
  captured mode rather than eager fallback, launched the reusable graph exec,
  polled its completion event, checked output bytes, and closed every retained
  resource. The separately executed residual specialization remained the eager
  control. This proves the private primitive graph lifecycle, not a complete
  model graph, graph-memory bound, shape matrix, or performance improvement.
- A complete tiny paged-BF16 graph then passed 128 capture-required cycles
  through the public `device` ordered executor. Each cycle ran twelve launches
  in Llama order: embedding, attention norm, QKV, positioned RoPE, mixed
  prefill/decode paged attention with split K/V writes, dense output plus
  residual, MLP norm/gated MLP/residual, final norm, and LM head. An independent
  MoonBit CPU referee checked fourteen intermediate/output boundaries, both KV
  arenas, every final logit, and the greedy token. The result was token `7`,
  maximum absolute error `0`, required capture selected, resources closed, and
  empty runtime stderr. This is complete synthetic graph correctness and
  lifecycle evidence, not an approved-model serving, shape-matrix, soak, or
  performance promotion.
- GPU memory was 15 MiB on GPU 0 and 16 MiB on GPU 1 both before and after the
  128-cycle probe. No compute application remained.
- CUDA and NCCL ABI boundary gates passed.
- Eight native sanitizer gates passed: CUDA ordered executor, CUDA peer access,
  NCCL, process child control, approved filesystem, TCP buffer alias,
  monotonic clock, and process maintenance. On this Linux/Clang 14 combination
  the probes use `setarch -R` to avoid nondeterministic ASan shadow-map
  collisions; the probe code is instrumented, but the MoonBit runtime is not.
- The numeric/I8 admission, loader/materializer, execution/bootstrap, release,
  runtime-instance, topology, rank, tensor-parallel, and sharded-materialization
  host gates passed.
- The static hostile-context OCI gate passed.

Selected remote evidence hashes:

```text
3770fc6672508682385f23004eaefe64fbe8d000ee328c4c95c94e357348b78c  moon-check-native-final.log
fed229ef6460fdc376a95d5fdd1a5668d3ee73700430466987b0038b  moon-test-native-final.log
b874336de54d0f88c6b33436f959e4d238e76efbb6e5e92e13d8e95162704084  physical-probe-128-final.log
967014dbac3285a1f8f75cb0e2f22d861ba07efecc898335451965071bf403fd  lunaflux-doctor.log
a475b8eaa74a656bbc1032fadfaac2df17802de3db4b98eb2bac568411296d6f  validate-cuda-abi.log
88e9bcf3db85e107dd8f93983a80ac8500ec62c2f691d6ecc79610fd2be9ab52  validate-nccl-abi.log
c723c3e2b69ed3d23f5af1a65c1b2f33dba9c6f8af83c1e38ba3431ad0510ce0  kernel-root/sha256/c723c3e2b69ed3d23f5af1a65c1b2f33dba9c6f8af83c1e38ba3431ad0510ce0.cubin
3388e15d1e247015dfae8537e002113278cf63e2ce74cd045100b7b80f88366b  evidence/luna-cuda-aot-evidence-v1.txt
c3f242acf5c995fafe3dc4fc09b81f476e7b4434d91124ccf7cc2a8720118c45  evidence/physical-residual-128.stdout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  evidence/physical-residual-128.stderr.log
437bc28e313c554cfedfe1b775a646413ad08c5599b632a3d7d87777c69989e1  paged-attention/generated_reference_v1_ep3.cu
9d5b37bdfc8429632c7819ca90922325bb6450fa063094951f11f9c43ea444ff  paged-attention/generated_reference_v1_ep3.recipe
1f8c2d43f5280c9f4244ed749204be121506069a59e1cc8f20c4d48a27b0dbd2  paged-attention/evidence/stdout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  paged-attention/evidence/stderr.log
883d60a6f6de5b51cafc00ea57788e72aa06dcef338cd8b6ecf5b6045eb29047  cuda-graph/source.tgz
bb1ff8b40875ba6cb130faca11f8434e6d6e33ed2cf14ae11e1facfe2c221380  cuda-graph/stdout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  cuda-graph/stderr.log
b008fe07a2f833d5d2f7101a681cb14fbfff7979d934d504be3947b03393406d  cuda-graph/RESULT.txt
273e3d4c170bdeec1207fd00aa76a82ae4c73faf8484965effc7f365633e06de  bf16-families/source.tgz
1d84f5b36a87898bdf0bdbf0c13680c6f4fddf4f1b1263e02d067ddb021a8b4f  bf16-families/stdout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  bf16-families/stderr.log
2b6ba1131689f0d0695f031e24dbf1c46e23b281954ea68bb9911c4d23881ebc  bf16-families/RESULT.txt
a94cafd03093de0b5d5890c54bd36309b329d23cd60d61eae2e296f91758a3bd  final-source/source.tgz
f03a104384be9b0576f7c4595afbcdfac8c130e7c4aaea1237085331b2ba459e  final-source/moon-test.log
8bbaee0bab967c42ea9475b87291ccf9523a1858e8a39d3f98cb9e4e65d1509d  final-source/final-gates-strict.log
256205c68d5a59f4eb6d68661100a784d4a4e39f4661e288e9721f9724710cd3  paged-bf16-graph/source-overlay.tgz
38fcdc9c2d790e003d9d4d91b2e3085f7548a36a14a35bc7a2b990b57bb826dc  paged-bf16-graph/evidence-rerun1/RESULT.txt
4a9f261c4b39c63bb91f652c04e8075615edb6343a7d45d42edfd3ff8602b5c2  paged-bf16-graph/evidence-rerun1/stdout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  paged-bf16-graph/evidence-rerun1/stderr.log
993e9fd0d12015d4636a1f9d86a3f32911c03b5ca4d5adecf60c275b33bd131e  final-current/source.tar.gz
74c6c2e7ed285906e47884e1bc09e3e61fe85df59321f8d4026fa08304c6453e  final-current/evidence/RESULT.txt
4a9f261c4b39c63bb91f652c04e8075615edb6343a7d45d42edfd3ff8602b5c2  final-current/evidence/graph-physical.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  final-current/evidence/graph-physical.stderr
8351d804617ab9a5ca542d51b76685eefa2820394417e7a9f8852c9b9db41ae8  lunaflux-final-8351d804617ab9a5.tar.gz
6030269b276032149a52fac9dd92c8f6ffd1d55bfe0971cf180ce23598e8c741  final-current-8351d804617ab9a5/evidence/RESULT.txt
0155b46b8c3769804033cbc7aa6b824c45e9b309e88636100ad380e242aa0f19  final-current-8351d804617ab9a5/evidence/moon-check.log
88dc043884c8db7b416d47941148b2c05c22937808510e6493eefb8dafd1276f  final-current-8351d804617ab9a5/evidence/moon-test.log
4a9f261c4b39c63bb91f652c04e8075615edb6343a7d45d42edfd3ff8602b5c2  final-current-8351d804617ab9a5/evidence/graph-physical.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  final-current-8351d804617ab9a5/evidence/graph-physical.stderr
4af8bb87602a2b4ff26f6c7a21d8ebb1f9db99181dfb199d884e9f53a9e478eb  final-current-8351d804617ab9a5/evidence/i8-sm120-rejection.result
a4fd490e5937ebbbdb4711e70dacd444b9c6a5daeeb5aa83051fb5a39182b98d  final-current-8351d804617ab9a5/evidence/i8-sm120-rejection.stdout
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  final-current-8351d804617ab9a5/evidence/i8-sm120-rejection.stderr
```

The paged-attention recipe hash above records the bytes used by that immutable
physical campaign. The subsequent non-circular ownership refactor left the
CUDA source hash unchanged and refreshed only the checked non-bindable recipe
fixture to
`8e3cdc25e8a41d64c20ac29cde60abd25f6ffb19177f9a0dd5348d23d852730c`.
No physical rerun is claimed for the refreshed recipe.

## Gates that could not run

- The aggregate `validate-boundaries.sh`, runtime-instance static wrapper, and
  tokenizer allocation wrapper could not complete their PCRE2 source scans on
  the remote host because its installed `ripgrep` lacks PCRE2 support. The
  identical final tree passed all three gates locally; the tokenizer wrapper's
  19 executable tests passed remotely before its source scan, and compatible
  CUDA, sanitizer, allocation, I8, and release subgates were rerun separately
  with strict failure propagation and passed.

- Broad/concurrent listener serving, positive I8 inference, task accuracy,
  memory improvement, throughput, and latency. The pinned tiny BF16 model now
  crosses both the production child and one bounded native loopback listener
  for the exact two-token request, but that narrow pass does not establish
  broad model quality or general serving readiness.
- Physical tensor parallelism and NCCL collectives: GPU 1 lacks BF16 and the
  pair has no peer access. The correct result for this topology is rejection,
  not a degraded two-GPU run.
- OCI image construction, SBOM/rootfs scan, and container execution: Docker is
  not installed on the host, and no approved release build context/artifacts
  were supplied.
- Comparison against vLLM and SGLang: there is no common approved model,
  tokenizer, kernel bundle, or workload to compare.

`lunaflux doctor` remains truthfully `readiness: false`. A release promotion
requires a supported production kernel/model bundle and, for Phase 7, a
homogeneous supported multi-GPU node with peer access.
