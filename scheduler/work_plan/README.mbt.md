# Pure execution work ranges

`prefill_query_count` is a total progress/budget/capacity function. Scheduler
selection uses it without importing models, kernels or device backends. It does
not reserve pages or change request ownership; those effects remain with the
scheduler and KV allocator.

`RowWork` is an allocation-free value describing packed query, cached-context
and KV-write ranges plus the observed output token row. Both submitted-plan and
native-frame device descriptor construction consume the same range calculation.
An absent output is represented in the host plan, never by overloading a GPU
descriptor offset. In particular, it does not remove KV writes.

This package does not itself eliminate GPU output-head work, change wire ABI,
choose attention kernels or change scheduling fairness. Those consumers must
retain row identity and effects when specializing an execution graph.
