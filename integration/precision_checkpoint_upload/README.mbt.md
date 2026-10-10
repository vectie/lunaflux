# Bounded checkpoint device upload

`CheckpointDeviceWeights` streams original numerical representations into compact
device allocations. `CheckpointDeviceIndexTables` separately streams exact I64
discrete tables into I32 device storage; integer metadata never masquerades as
floating-point precision IR.

The table owner checks all device capacities and its source/output chunks plus
one uniqueness row per table before allocation. `i64_table_host_bytes()` budgets
these payload buffers, not reader authentication/header/index metadata or total
process memory. Input chunks may split any element or row. Range and within-row
uniqueness checks share the whole-table narrowing law. Partially uploaded device
bytes remain owned but cannot be exposed until every stream finishes. Explicit
close handles both successful and failed preparation, and can be retried after a
native release failure.

These are startup effects only. Prepared decoder queues borrow the resulting
allocations without filesystem operations, narrowing, validation scans or host
table copies during token execution. Close borrowing queues before table owners.

Native fixtures cover split chunks, exact bytes, pre-allocation budgets,
cross-chunk duplicates/range errors and balanced cleanup. The real-driver
`cmd/checkpoint_index_probe` exercises the same production loader with tiny
checkpoint files; it is not whole-model execution or a throughput benchmark.

Weight and narrowed-table staging chunks are discarded without release-time
scrubbing. Their valid byte counts bound every read and device copy; discarded
chunks cannot be borrowed again. Explicit device release and failed-upload
ownership are unchanged. Inventory labels do not verify payload integrity.
