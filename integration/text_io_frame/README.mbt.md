# Prepared text frame

`TextIoFrame` owns normalized/logit buffers and four AOT functions. Its borrowed
prefix and suffix launch views surround any compatible decoder frame in one
caller-owned queue, with one completion per step. Preparation binds all pointers
and checks workspace capacity; warmed launch execution allocates no heap memory.

Token IDs, input counts, selected row IDs/output count, residual input/output,
sampled token output and weights belong to the caller. Distinct residual input
and output allocations must outlive the borrowing queue. Validate request ports
before submission and reject sampled `-1`/decoder error ports before publishing.
Close/abort the queue before closing this frame or releasing borrowed buffers.
Partial preparation is releasable; repeated close is harmless.
