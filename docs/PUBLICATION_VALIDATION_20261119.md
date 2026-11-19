# Source publication validation — 19 November 2026

Pending changes were committed by compiler migration, model/media adapters,
device/kernel execution, serving, qualification tests, operational scripts,
configuration interfaces and documentation. Model weights and generated
`target/` artifacts remain local and excluded from Git.

The complete Linux AArch64 native run passed **5,544 tests with zero failures**
using the matched MoonBit v0.10.15 toolchain and copied pinned dependencies in
a container with networking disabled. Existing optional asset/device tests
print explicit skips; this is a software result, without a new physical CUDA,
FPGA, NCCL, full-model or deployment claim.

The preceding macOS run passed 5,552 of 5,554 tests; two actual-shape checkpoint
fixture writes failed because the local disk ran out of space. The full run
was moved to Linux with enough storage. Fixture sizes and assertions were not
reduced. Linux then exposed a callback-retirement test race: receiving the
HTTP response is earlier than server callback cleanup. The test now waits for
service readiness and asserts the same exact retirement count within its
existing timeout. The local three-test service regression and full Linux run
both pass.

`moon info` and `moon fmt` passed; generated interfaces reflect the explicit
derived-method promotions. The warning-denied native check remains open: its
last attempt reported 34 diagnostic errors from remaining compiler migration
warnings. Warning flags were not disabled, and a functional native test pass
does not claim that gate passed.

The [machine-readable record](evidence/publication-validation-20261119.json)
contains source revision, log hashes, skip count and environment identity.
