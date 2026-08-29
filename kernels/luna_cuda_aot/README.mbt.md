# Offline LunaTile CUDA AOT production

`kernels/luna_cuda_aot` lowers the one currently proved LunaTile semantic—an
exact BF16 residual add—into deterministic CUDA source and a closed CUBIN
compilation recipe. Its ABI is exactly three pointer operands in launch-contract
order: left activation, right activation, output activation. The exact element
count is baked into the specialization, so no scalar-by-value argument crosses
the production ABI. The lowering also publishes exact one-dimensional launch
dimensions and BF16 operand byte counts for release-side launch-contract
construction; the grid covers every baked element using a fixed 256-thread
block.

The package never opens a process, filesystem, CUDA context, module, compiler,
or request. An isolated release builder invokes the offline wrapper, compiles
the same source twice, and returns two immutable CUBIN snapshots plus a narrow
receipt. Admission recomputes both hashes, requires byte equality, binds the
typed source, recipe, target, externally approved complete-toolchain manifest
digest and generated symbol. A second join requires the exact admitted
specialization to name the resulting module, program, AOT plan, compiler
policy, execution shape, and entry-point identity before it publishes an
existing `KernelModuleInput`/`KernelEntryPointInput` pair. Production runtime
continues to consume only digest-admitted artifacts through
`kernels/artifact`; it has no dependency on this offline package.

The release module path is always
`sha256/<artifact-sha256>.cubin` relative to the approved kernel root. Offline
build evidence is a separately inventoried release-evidence object and is not
placed beside runtime modules unless the release schema explicitly permits it.
No caller-provided source fragment, compiler flag, output path, or CUDA symbol
is accepted by the MoonBit lowering boundary.

The wrapper separately inventories the invoked `nvcc` and `ptxas` binaries for
diagnostics, but does not misrepresent those two files as the complete CUDA
toolchain. The recipe's `toolchain_sha256` is the digest of a caller-supplied,
externally approved immutable manifest covering the complete compiler image,
headers, device libraries, and delegated tools. That approved manifest must
also contain the exact diagnostic driver-identity digest, so the actually
invoked `nvcc` and `ptxas` pair cannot be substituted under an otherwise
approved toolkit claim.
