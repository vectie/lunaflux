# GLM-5.3 host materialization

This package is the startup-only bridge from exact GLM-5.3 safetensors shards
to bounded final host arenas. It admits only the converted mixed BF16/F32
logical manifest. Official block-FP8 remains a typed unsupported encoding.

Header parsing, duplicate-key rejection, file authentication, stamp replay,
range validation, and approved-file close handling are delegated to the
family-neutral `model/streaming_safetensors` package. GLM retains only its
manifest/numeric admission, exact BF16/F32 role checks, payload-coverage rule,
canonical arena layout, and release contract.

Payload bytes are read from pinned approved files directly into final segmented
arenas with one neutral multi-destination copy call. Names, shapes, dtypes,
byte ranges, shard content digests,
manifest/execution identity, numeric schema digest, and numeric binding digest
must all agree before an open owner is returned.

`Glm53HostWeights::close` deterministically zeroizes and invalidates every
arena and drops its references. MoonBit `FixedArray` storage is GC-managed, so
the package deliberately does not claim deterministic OS-level deallocation,
device materialization, kernel readiness, or execution authority.
