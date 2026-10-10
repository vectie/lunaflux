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
