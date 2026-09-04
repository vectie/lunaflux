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

The semantic program remains unchanged and authoritative. CUDA, HIP, Metal,
and CPU lowerings may interpret the same selected regions differently. Device
probing, benchmarking, cache lookup, and source publication are intentionally
outside this package.
