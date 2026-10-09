# Two-Spark execution workstream

Branch: `twospark`. The branch implements the two-node compiler, rank-local
admission, authenticated control sessions, the existing rank execution
protocol, worker-service/online composition, recovery receipts, and an
operator-launched rank executable. Host tests exercise these connections.
Physical two-Spark execution and performance are **not qualified**. The pinned
TLS server dependency also needs resolution before production release.

## Architecture

Two explicitly assigned DGX Sparks connected through ConnectX-7 form one
LunaFlux instance: one scheduler and two rank workers. This is a post-v1
capability; it does not change the same-host admission boundary of Phase 7.
Deployment owns process launch, local files and endpoints. LunaFlux does not
add SSH, artifact distribution, fleet placement or automatic discovery.

The functional compiler remains:

`immutable model plan → family-owned TP lowering → pure two-node capacity join
→ rank-local device/KV/AOT plans → explicitly owned execution resources`.

Model-family choices stay in model builders, hardware choices in device/kernel
packages. The scheduler, KV allocator and public request schema acquire no
network/model/backend branches. Native CUDA/NCCL interfaces and AOT execution
are reused. No production Python or runtime JIT is introduced.

Node identity qualifies device ordinal: node A/device 0 and node B/device 0
are distinct devices. The two memory pools remain separate. Each rank must fit
its own system reserve and runtime ceiling; logical KV capacity is bounded by
the smaller aligned rank allocation. Declared NICs and budgets are not evidence
of observed RDMA bandwidth or available physical memory.

## Implemented paths

- `device/network_topology` and `engine/two_node_plan`: immutable two-node
  topology, complementary shard plans, collective order, per-rank budgets,
  aligned KV capacity and a canonical startup digest.
- `engine/remote_startup`: root-free 1112-byte rank Configure contracts binding
  both workers, deployment/executable digests, group generation, lease and
  NCCL rendezvous. A contract value alone is not channel authentication.
- `tensor_parallel_rank_child.compile_remote_plan/load_remote_rank_plan`:
  independently reconstruct the plan from local approved model/AOT roots and
  enter the existing device worker. Node/NIC environment admission occurs
  before GPU resources are acquired.
- `runtime/remote_tls`: bounded full-duplex TLS frames with hostname/CA
  verification and a startup credential bound to the exact rank contract.
  `runtime/remote_session` drives heartbeat/acknowledgment and a graceful
  goodbye handshake. The older raw TCP channel remains a transport primitive;
  it is not used as authenticated serving transport.
- `runtime/remote_mailbox`, `engine/remote_group`, `rank_child_control`:
  fixed mailboxes carry the existing Configure/Ready, Submit/Poll/Complete,
  graph sidecar, Drain/Close protocol. Completion requires both ranks and the
  original rank-group semantic owner validates scheduler retirement.
- `engine/remote_rank_worker`: reuses the admitted rank child execution
  machine. Authenticated heartbeats renew the actual device owner's local
  lease; idle control turns also poll it. Transport/lease failure aborts the
  communicator and releases resources through the existing owner.
- `engine/worker_service.prepare_owned_remote` and
  `service/online_session.prepare_remote_luna_online_instance`: publish the
  existing service only after both ranks are Ready. `.framed(...)` joins the
  existing tokenizer preparation pool, framed coordinator and native/OpenAI
  server constructors. Public request behavior is unchanged.
- `engine/remote_release`: a separate authenticated recovery connection can
  acknowledge only a closed worker. Service recovery waits for both receipts,
  retires scheduler work, invalidates device state and activates a complete
  fresh group. Failed replacement resources have their own receipt obligation;
  their generations cannot be reused.

## Coordinator composition

Use the two-node compiler and `tensor_parallel_worker_bootstrap.admit_network`
to construct the group and its two rank envelopes. Prepare the existing service
through `prepare_owned_remote` (or the online/framed preparation), then connect
each rank with `remote_tls.connect`. Bind each connection to the serialized
`RemoteRankStartup`, use the same wire limit on both ends, and set TLS capacity
to `RankGroupWireLimits.max_frame_bytes() + 1` for the session tag.

`remote_coordinator.with_group` owns the two sessions around a
generation-scoped async body. That body polls the pending preparation to Ready
and drives normal service progress and cooperative shutdown maintenance. It
must return after the group is Closed. A failed session cancels this body and
invalidates the group. An API loop intended to survive replacement must live
outside that cancellation scope; catch the generation driver's failure and
drive the existing service recovery maintenance from the retained owner.
If scheduler publication is backpressured after a completion was already
accepted, drain output and finish that exact service commit first. A concurrent
disconnect preserves accepted-result retirement and publishes an idle group
failure after commit; it cannot revoke already completed computation.

Recovery uses fresh TLS connections and `remote_release.receive`; pass the
receipts to `RemoteWorkerServiceControl.confirm_release`. After scheduler
retirement/invalidation, `prepare_replacement` returns the new group for its
own sessions and Ready handshake. Then the service's normal restart transition
activates it. If that attempt fails, use `confirm_replacement_release` and
replacement cleanup maintenance before another attempt. Socket closure or
lease expiry alone never proves native cleanup.

A dead process cannot supply a receipt. Deployment must retire the instance
when release cannot be established; this branch does not implement external
process-kill attestations or reconnect an old generation into execution.

These are framework composition APIs. The standard `lunaflux run` deployment
descriptor does not automatically select this experimental remote route.

## Rank executable and deployment contract

Build `cmd/two_spark_worker` for native ARM64 on the target toolchain. Start one
process on each node with:

```text
two_spark_worker ABSOLUTE_DEPLOYMENT_ROOT CONFIG_SHA256
```

The root contains `two-spark.json` (at most 64 KiB), whose complete bytes must
match `CONFIG_SHA256`. Its exact schema is `lunaflux.two-spark-worker.v1`:

| Field | Value |
| --- | --- |
| `schema` | `"lunaflux.two-spark-worker.v1"` |
| `source` | Pinned encoded bootstrap source |
| `workers` | Two pinned existing worker startup frames, rank order |
| `envelope` | Pinned rank-local TP envelope frame |
| `nodes` | Two `{node, device_ordinal, physical_memory_bytes, rdma_interface}` objects |
| `budgets` | Two `{system_reserved_bytes, runtime_ceiling_bytes, activation_workspace_bytes, collective_bytes, kv_bytes}` objects |
| `executables` | Two executable SHA-256 strings, rank order |
| `deployment_sha256` | Deployment identity SHA-256 |
| `rank` | `"0"` or `"1"` |
| `lease_millis` | Decimal string, 30–300000 |
| `model_root`, `kernel_root` | Absolute node-local approved roots |
| `listen` | Socket address, e.g. `"192.0.2.1:9443"` |
| `certificate`, `private_key`, `credential` | Pinned PEM certificate, PEM key and exactly 32 credential bytes |
| `max_payload_bytes` | Decimal string, at least 1112 and enough for existing worker frames |

Every pinned reference is `{ "path": "relative/path", "sha256": "64 hex characters" }`.
All numeric fields, including nested ordinals/budgets, are decimal **strings**
to preserve exact 64-bit values. The source and envelopes use the existing
binary encoders; they are not JSON translations. Deployment pins the running
executable and supplies the matching full model files/AOT artifacts locally;
the worker uploads only its assigned shard. The rank entry currently uses the
existing `source.execution()` bootstrap route; other quantized bootstrap
variants require their corresponding explicit admission route.

Before launch set `LUNAFLUX_NODE_ID` to the declared node, `NCCL_NET=IB`,
`NCCL_IB_DISABLE=0`, and `NCCL_IB_HCA` to `=` followed by the declared interface.
This validates configuration, not successful hardware transport selection.
Start/connect both ranks within the lease window. After a session ends, the
process closes its worker and retains a release-only listener at the same
address. A new generation requires fresh processes; deployment retires old
listeners after collecting receipts.

## Performance and release limits

Artifact hashing, certificate/credential admission and canonical deployment
checks occur during startup. The scheduler uses preallocated mailboxes and a
reusable monotonic clock reader. GPU execution reuses the existing executor;
there is no per-token artifact hashing or filesystem validation.

TLS framing still encrypts and copies control messages, and async mailbox
drivers currently yield/poll at 1 ms when idle. These have real latency and CPU
costs even though they are outside the synchronous scheduler. No zero-overhead
or remote allocation-free claim is made. Measure TTFT, inter-token latency,
collective time and throughput before adding more checks or optimizing.

The pinned `moonbitlang/async` TLS server entry point is marked internal and
for testing by its provider. Host tests use it, but this does **not** qualify it
as a supported production TLS server. Production release requires a supported
provider/API decision. Test keys under `runtime/remote_tls/test_keys` are public
fixtures and must not be deployed.

## Validation and machine gates

On 2026-09-26, the final complete native run passes **4083/4083 tests**;
the final targeted run passes **167/167**. The rank executable builds and its
captured usage diagnostic is verified. Both existing release-mode warmed
allocation executables (TP device worker and local rank-child control) pass.
Formatting and scoped interface generation pass. Core, rank-child control,
TP rank-child, TP device-worker, KV-plan and online-TCP boundary checks pass.
These allocation results cover the existing local execution paths, not the
async remote transport. The final logs are
`/tmp/lunaflux-remote-all-current.log`,
`/tmp/lunaflux-remote-final-current.log` and
`/tmp/lunaflux-remote-entry-build.log`.

Warning-denied root check/test stop on existing warning-79 migration errors
(142/178 diagnostics in these runs). The aggregate service-boundary script
also retains failures in existing claim/framed-surface expectations and scans
of generated integration build copies; its online-TCP sub-gate passes after
admitting exactly the two new private-buffer transport callers. These are not
reported as clean release gates, and no warning suppression was added.

Host tests cover pure capacity/startup admission, real loopback TLS, idle lease
renewal, exact framing, both-rank startup/close, synthetic prefill/decode through
the real scheduler/service, and recovery/replacement receipt ownership. Mock
rank tokens test protocol composition, not inference numerics. Strict root
validation also encounters pre-existing MoonBit warning-79 migration issues;
the TLS provider emits its internal-API warning. No warning suppression is
added to hide these limitations.

When the machines arrive:

- Pin ARM64 runtime, CUDA/NCCL, `sm121` AOT/model/tokenizer artifacts and NICs;
  verify actual NCCL RDMA transport selection and measure both directions.
- Compare single-node and two-node logits/tokens, then run a supported model
  exceeding one node's weight budget.
- Exercise prefill/decode, mixed batches, prefix reuse, cancellation,
  backpressure, rank/cable loss, drain and fresh-group recovery.
- Run required native sanitizer, numerical, resource-balance and allocation
  gates, then benchmark TTFT, inter-token latency, throughput and memory.

Pipeline/expert parallelism, arbitrary cluster sizes, new quantization formats
and fleet automation are outside this milestone.
