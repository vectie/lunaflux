# Row projection normalization AOT lowering

`RowProjectionNormPrecision` supplies immutable shape, packed matrix and numeric
contracts. CUDA lowering composes the shared projection and weighted RMSNorm
source families into two pointer-only ABI functions. Source generation occurs
before AOT compilation, never on the inference request path.

The native exporter is activated by `LUNA_EXPORT_ROW_PROJECTION_NORM=1`.
`scripts/run-row-projection-norm-probe.mbtx` exports this exact renderer, compiles
and runs a small independent C++ numerical oracle on GB10, then runs memcheck,
racecheck and synccheck and downloads the executable with matching SHA-256.
Its component fixture is not full-checkpoint or serving performance evidence.
