# Descriptor-backed runtime report

This native-only operator package projects an already-admitted
`RuntimeDescriptorAdmission` into fixed-order memory and capacity evidence. It
does not open files, probe a device, allocate device storage, or make a
readiness decision.

The reported reserved device total is checked `Int64` arithmetic over exactly:

`weight reserved arena + activation/workspace reserved arena + KV reserved arena`

Workspace is a subregion of the activation/workspace arena and is never added
again. Kernel module file bytes are artifact evidence and are never folded into
the device allocation total. Activation bytes are labelled as a reserved
high-water mark; activation region logical sizes are not additive because the
planner reuses slots across non-overlapping lifetimes.

CUDA-graph memory is a separate authenticated declared startup upper bound,
not an observed allocation. A bounded value is reported only when a
capture-safe Phase-5 authorization declares a positive aligned bound and the
strict runtime descriptor supplies a positive ceiling that contains it. The
graph-aware device startup upper bound adds that declared bound exactly once
to the arena total. Legacy eager-only descriptors report graph accounting as
absent, even when inert authenticated metadata carries a future capture bound.
A v2 ceiling paired with absent or capture-unsupported metadata is rejected at
startup instead of being rendered as zero bytes. Kernel inspection retains the
separate explicit unsupported diagnostic for authenticated capability
metadata.

Descriptor v1 does not admit scheduler, cache, or service configuration. The
report therefore labels those capacities unavailable instead of deriving them
from worker execution ceilings.
