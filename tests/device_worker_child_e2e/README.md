# Device worker child startup gate

This CPU-linked end-to-end gate spawns the real startup-only device child with
fixed inherited model and kernel roots, sends Configure followed by the exact
canonical source, and supplies a digest-correct but unsupported model config.
It proves the real composition emits no Ready and exits silently with status 1
before CUDA is opened. A second spawn sends an unexpected source-slot frame and
proves the command again emits no Ready and exits status 1. Physical successful
readiness is intentionally deferred.
