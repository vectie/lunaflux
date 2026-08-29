#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
package="$root/engine/fp8_device_executor"
interface="$package/pkg.generated.mbti"

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file")
  [ "$lines" -lt 500 ] || {
    echo "FP8 device executor source exceeds 499 lines: $file" >&2
    exit 1
  }
done

rg -q 'fp8_release_authority' "$package/moon.pkg"
if rg -q 'luna_kernel_bundle|LunaDeterministicCompileReceipt' \
  "$package" --glob '*.mbt' --glob 'moon.pkg'; then
  echo "caller-constructible compile receipts crossed executable boundary" >&2
  exit 1
fi
if rg -q 'cmd/|engine/device_worker|engine/worker|service/|runtime/descriptor' "$package/moon.pkg"; then
  echo "FP8 executor was activated from a product/runtime route" >&2
  exit 1
fi

rg -q 'pub fn prepare_v2\(' "$interface"
rg -q '@fp8_release_authority.Fp8ReleaseAuthorityV2' "$interface"
if rg -q 'pub fn prepare_v2\([^)]*Bytes|pub fn prepare_v2\([^)]*LunaKernelCompiledOperation' "$interface"; then
  echo "public prepare accepts unauthenticated module bytes" >&2
  exit 1
fi
if rg -q 'pub (struct|enum).*(@device\.(Module|Function|Stream|Allocation|KernelArguments))' "$interface"; then
  echo "private native resource escaped the executor owner" >&2
  exit 1
fi

rg -q 'copy_to_fixed_host' "$package/execute.mbt"
rg -q 'admit_fp8_execution_evidence' "$package/execute.mbt"
rg -q 'Fp8DeviceFailure\(ScaleReadback' "$package/execute.mbt"
rg -q 'ScaleEvidence' "$package/execute.mbt"
rg -q 'byte_count != 4L && byte_count != 8L' "$package/scales.mbt"
rg -q '0x7fc00000U' "$package/scales_wbtest.mbt"

if rg -q 'copy_to_host\(' "$package/execute.mbt"; then
  echo "token-step scale/output readback allocates a fresh host buffer" >&2
  exit 1
fi
if rg -q 'allocation\.close\(\) catch' "$package/prepare.mbt"; then
  echo "construction bypasses retryable aggregate cleanup" >&2
  exit 1
fi
if rg -q 'Array::|let [^:]+ : Array\[|FixedArray::make|Bytes::' \
  "$package/execute.mbt"; then
  echo "token-step path contains managed storage construction" >&2
  exit 1
fi

echo "FP8 device executor boundary validation passed"
