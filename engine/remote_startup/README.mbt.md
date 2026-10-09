# Root-free two-node startup envelopes

`RemoteGroupStartup::new` composes an immutable pair of rank messages from one
compiled two-node plan, two existing worker startup contracts, two executable
digests, a deployment digest, a group generation, an NCCL rendezvous identity,
and a bounded lease duration. Each local worker contract retains its own
bootstrap/source identity; model generation, predecessor and runtime limits
must agree. Node-local ordinals are checked against the explicit plan.

No filesystem descriptors, approved roots, absolute paths, credentials, or
remote timestamps cross this schema. The deployment digest is a content pin,
not a signature or authentication credential. A group digest covers both
rank contracts, preventing one node's executable/source replacement from
silently leaving the other node's expectation unchanged.

The v1 Configure envelope is exactly 1,112 bytes, little-endian:

| Offset | Bytes | Contents |
| --- | --- | --- |
| 0 | 64 | Magic `0x5253464c`, version 1, frame length, rank, generation, label lengths, world 2, worker length 408, RoCE code 1, zero reserved fields, lease duration |
| 64 | 64 | Complete two-node plan SHA-256, lowercase ASCII |
| 128 | 64 | Deployment SHA-256 |
| 192 | 64 | Both-rank group SHA-256 |
| 256 | 64 | This rank's executable SHA-256 |
| 320 | 128 | ASCII node label with zero padding |
| 448 | 64 | ASCII RDMA interface label with zero padding |
| 512 | 128 | Existing opaque NCCL rendezvous bytes |
| 640 | 408 | Existing worker-wire Configure v4 |
| 1048 | 64 | Domain-separated frame SHA-256 |

Group hashing uses domain `lunaflux.remote-group-startup.v1` followed by each
rank's first 1,048 bytes in rank order, with the group-digest slots still zero.
Frame hashing uses domain `lunaflux.remote-rank-configure.v1` followed by its
first 1,048 bytes after filling the group digest. Changing either embedded
schema requires an explicit version update; no fallback decoding is provided.

`copy_to` fills caller-owned fixed storage. `verify_configure` checks the exact
length and every byte against an independently composed expectation, including
padding and reserved fields. It never adopts an incoming frame as its own
expectation. These operations are for startup only, with no retained alias to
caller storage. The package does not listen, authenticate a peer, start NCCL,
or send Ready. Those are remaining remote transport/execution-owner work.

`start_lease` creates the value-only local watchdog. The future resource owner
must enforce its terminal state independently of NCCL progress and must fence
the old generation before a replacement can become Ready.
