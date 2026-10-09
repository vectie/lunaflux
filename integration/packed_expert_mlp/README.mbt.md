# Executable compact expert program

bind prepares reusable ordered launches from the AOT source and caller-owned
device functions/allocations. Submit them on one stream through the existing
OrderedKernelExecutor; completion and deterministic release stay with its owner.
No launch-path file inspection, format switch, compilation or allocation occurs.

Routing arrays contain unique global expert IDs per row. The immutable map from
ExpertMlpPrecision maps them to local bank positions, or -1 for another rank.
Unowned contributions become zero; the result is rank-local F32. A distributed
caller must reduce contributions and apply its model's final output rounding.
