# Functional attention tile optimizer

This package recognizes backend-neutral equational regions in immutable
attention tile programs. Its bounded pass list is data, pass application is a
pure fold, and the result contains both an extraction plan and a deterministic
trace.

The semantic program remains unchanged and authoritative. CUDA, HIP, Metal,
and CPU lowerings may interpret the same selected regions differently. Device
probing, benchmarking, cache lookup, and source publication are intentionally
outside this package.
