# Phase-7 physical topology diagnostic

This package is LunaFlux's product-owned physical admission runner. It composes
the existing CUDA inventory, every directed peer-access query, the strict
homogeneous full-mesh local topology admission, and the dynamically loaded
collective-runtime admission into one immutable, authority-free schema-v1
evidence value. The transient collective-runtime admission is used only to
authenticate its exact version and is not retained.

Expected unsupported machines return `PhysicalTopologyRejected` evidence.
They do not raise from the command, enable peer access, create a CUDA context,
allocate device memory, construct a communicator, spawn a rank, or silently
degrade tensor parallelism. Missing `libnccl.so.2` is the typed
`CollectiveRuntimeLibraryMissing` observation. If physical topology otherwise
admits, that observation becomes `CollectiveUnavailable`.

The renderer has fixed field order. Exact device names are encoded as UTF-8
hex so a vendor string cannot inject evidence fields. The standalone command
prints only this schema; an unsupported topology is a successful diagnostic
execution with `outcome: rejected`.
