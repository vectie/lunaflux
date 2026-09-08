# Functional attention tile optimizer

This package recognizes backend-neutral equational regions in immutable
attention tile programs. Its bounded pass list is data, pass application is a
pure fold, and the result contains both an extraction plan and a deterministic
trace.

The first common-subexpression pass recognizes that paged K/V vectors from
one logical key row share the same page-table lookup. It records a pure hoist
region across the key and value streams. A backend may realize that region
with subgroup broadcast, shared storage, or scalar reuse; the optimizer does
not name the mechanism.

The liveness pass also recognizes the matrix-QK/PV online-softmax pipeline:
the staged key tile is dead before the probability tile is consumed, while
maximum and denominator are query-local fold values. It records storage reuse
as a portable semantic fact. It does not choose registers or local memory.

The shared-key pass interchanges an independent query map with the dot's
ordered reduction. A product of query accumulators consumes one common key
operand at each reduction position. Each dot keeps its accumulation order;
this is not floating-point reassociation. Terminal backends decide whether
their matrix granularity can realize the sharing profitably.

The semantic program remains unchanged and authoritative. CUDA, HIP, Metal,
and CPU lowerings may interpret the same selected regions differently. Device
probing, benchmarking, cache lookup, and source publication are intentionally
outside this package.
