# Root-free two-node startup envelopes

`RemoteGroupStartup::new` composes an immutable pair of rank messages from one
compiled two-node plan, two existing worker startup contracts, two executable
digests, a deployment digest, a group generation, an NCCL rendezvous identity,
and a bounded lease duration. Each local worker contract retains its own
bootstrap/source identity; model generation, predecessor and runtime limits
must agree. Node-local ordinals are checked against the explicit plan.

No filesystem descriptors, approved roots, absolute paths, credentials, or
remote timestamps cross this schema. The deployment and executable fields are
supplied association labels, not verified content pins. The legacy group-digest
field reuses the deployment label without hashing either frame. Each rank's
contract is compared explicitly; a peer's executable change does not create a
new checksum for the local rank. Generation and rendezvous are separate fields.

The v1 Configure envelope is exactly 1,112 bytes, little-endian:

| Offset | Bytes | Contents |
| --- | --- | --- |
| 0 | 64 | Magic `0x5253464c`, version 1, frame length, rank, generation, label lengths, world 2, worker length 408, RoCE code 1, zero reserved fields, lease duration |
| 64 | 64 | Existing two-node plan identity, lowercase ASCII |
| 128 | 64 | Supplied deployment label |
| 192 | 64 | Same supplied deployment label (legacy group-digest slot) |
| 256 | 64 | Supplied rank executable label |
| 320 | 128 | ASCII node label with zero padding |
| 448 | 64 | ASCII RDMA interface label with zero padding |
| 512 | 128 | Existing opaque NCCL rendezvous bytes |
| 640 | 408 | Existing worker-wire Configure v4 |
| 1048 | 64 | Unused legacy checksum tail; zero on send, ignored on receive |

Startup performs no group/frame hashing. The embedded worker v4 checksum word
at outer offsets 976–979 is also ignored, matching the worker codec. Header,
rank, generation, rendezvous, node/NIC and worker-contract comparison remain.
The frame size stays fixed for compatibility; update both endpoints together,
because old checksum-producing peers compose different expected bytes. The
label is not unique to generations: replacement uses monotonic generation and
retirement, never label inequality as a lifecycle prerequisite.

`copy_to` fills caller-owned fixed storage. `verify_configure` checks the exact
length and contract bytes against an independently composed expectation,
excluding unused checksum fields. It never adopts an incoming frame as its own
expectation. These operations are for startup only, with no retained alias to
caller storage. The package does not listen, authenticate a peer, start NCCL,
or send Ready. Transport/execution owners retain those separate responsibilities.

`start_lease` creates the value-only local watchdog. The future resource owner
must enforce its terminal state independently of NCCL progress and must fence
the old generation before a replacement can become Ready.
