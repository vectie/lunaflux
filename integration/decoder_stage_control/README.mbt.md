# Plain decoder-stage control

The coordinator sends the existing canonical plan, waits for metadata DMA, asks
the rank to execute, receives its execution result/token, then commits or
poisons the whole step. A partial rank aborts its local completion writer; it
never publishes a model response. Only a terminal rank supplies sampled output.

Activation bytes travel separately. `arm_activation` starts the receiver after
rank metadata retirement; `ready_to_submit` must remain false until its upload
retires. The server's queue cannot run before that condition. The enclosing
worker polls both owners and drains the activation edge before closing its rank.

Control storage is fixed and bounded. The reported ceiling covers byte/integer
buffers; reserve socket/object runtime overhead, rank ports and activation DMA
separately. Step epochs, row semantics and commit preserve execution ownership;
this is not TLS, authenticated transport, RDMA or fleet orchestration.

Idle between requests has no frame deadline. Once a first byte arrives, the
frame deadline is absolute; metadata/activation/execution also have bounded
phase deadlines. A failed/poisoned rank is not silently reused.

After an acknowledged request release, an orderly EOF at the next frame boundary
terminates the server cleanly. Commit alone does not release request ownership:
EOF before release or within any next prefix/body remains a failure. A released
connection can also accept another prepare, preserving reusable-rank behavior.

An optional startup-bound committed side effect can consume a retired base
step. It runs only after a successful whole-pipeline commit; failed commits skip
it. Its completion is polled before commit acknowledgement or new metadata is
accepted. Prefill/decode kind comes from the validated plan, not row-count
guessing. The default route has no extra side effect or wait phase.

## Tentative verification

An explicitly configured terminal rank/client can return every live output row.
Results stay in the fixed control buffer until commit/rollback; scalar getters
reject a multi-row vector instead of truncating it. Ordinary one-token messages
retain their existing format. Multi-output peers must both select the matching
capacity; failure exposes no partial vector.

Prepare flag `1` requests a tentative state transaction. The server rejects it
unless startup supplied `StageStateTransaction`. Save completion precedes input
acknowledgement, activation arming and model execution. Command `9` requests
rollback; receipt `10` arrives only after the complete device restore and the
rank's host-frontier rollback retire. Rejection does not run the committed
prediction side effect. An all-accepted commit discards the backup; a failed
commit also releases that backup after retirement but permanently poisons the
rank and never publishes successful output. Both continue to use plaintext,
bounded control transport, without payload hashes.

The state-effect owner must snapshot all actual mutable allocations, not just a
logical length. It separately budgets backup bytes and drains/cancels model work
before resolving an abandoned save/restore and closing snapshot leases. Closing
the server closes transport only; it does not make unresolved tentative device
state committed or release the borrowed rank. Native TCP tests cover save/restore
ordering, every returned row, replay of an accepted prefix, failed later samples
and unchanged one-token behavior. They do not prove a checkpoint DSpark loop.
