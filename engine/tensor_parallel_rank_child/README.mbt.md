# Tensor-parallel rank child

This package owns the production inherited rank-control boundary and local
rank bootstrap. It receives the canonical typed Configure into bounded fixed
storage and authenticates it against the outer wire binding before opening any
root or device authority.

Bootstrap reopens the exact inherited model and kernel roots, digest-loads the
model configuration, rebuilds the model plan, probes and admits the declared
local topology, and inspects only this rank's sharded weight slice. The shared
v4 source-policy mapper derives the full admitted runtime and KV plan. The
rank-local manifest loader and device-worker admission then rederive and
authenticate the device, execution, collective, KV, artifact, generation, and
runtime-version evidence in the rank envelope.

`Ready` is published only after the real tensor-parallel device worker has
prepared every resource and completed collective startup. The bounded control
loop accepts canonical plan frames, progresses actual rank execution, publishes
leader completion bytes or follower scalar acknowledgements, reports typed
collective/rank failures, and performs Abort/Drain/Close cleanup before the
terminal response. No digest is treated as authority and no fixture or echo
worker can cross the readiness barrier.
