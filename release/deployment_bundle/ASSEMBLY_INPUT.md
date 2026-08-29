# Assembly input v1

`scripts/assemble-release-bundle.sh` accepts one absolute input path with an
independent `#sha256=` suffix and one new absolute output path. The input is
exactly 27 newline-terminated `key=value` lines in this order:

~~~text
schema=lunaflux.deployment-assembly.v1
base_image=<untagged repository>@sha256:<64 lowercase hex>
linux_architecture=x86_64|aarch64
lunaflux_source=<canonical absolute regular file>
lunaflux_sha256=<64 lowercase hex>
worker_source=<canonical absolute regular file>
worker_sha256=<64 lowercase hex>
launch_source=<canonical absolute regular file>
launch_sha256=<64 lowercase hex>
model_source_root=<canonical absolute directory>
model_inventory_source=<canonical absolute regular file>
model_inventory_sha256=<64 lowercase hex>
runtime_descriptor_relative=<strict relative JSON locator>
runtime_descriptor_sha256=<64 lowercase hex>
policy_source_root=<canonical absolute directory>
policy_inventory_source=<canonical absolute regular file>
policy_inventory_sha256=<64 lowercase hex>
instance_policy_relative=<strict relative JSON locator>
instance_policy_sha256=<64 lowercase hex>
kernel_source_root=<canonical absolute directory>
kernel_inventory_source=<canonical absolute regular file>
kernel_inventory_sha256=<64 lowercase hex>
kernel_manifest_relative=<strict relative JSON locator>
kernel_manifest_sha256=<64 lowercase hex>
runtime_library_source_root=none|<canonical absolute directory>
runtime_library_inventory_source=<canonical absolute regular file>
runtime_library_inventory_sha256=<64 lowercase hex>
~~~

Each non-library inventory is a sorted, exact, nonempty `sha256sum` file with
strict-relative paths. Model payloads are limited to JSON and safetensors;
policy payloads to JSON; and the kernel root to its sole JSON execution
manifest plus `.cubin`, `.fatbin`, or `.bin` AOT modules. A `none` library root
uses an inventory whose exact content is `none` plus a newline. Otherwise only
real `.so` payloads are accepted; symlink aliases are forbidden.

The output is claimed with an atomic new-directory creation and is removed on
failure. Existing output is never changed. A complete result contains separate
read-only launch/model/policy roots, `oci-context/`, canonical root-free
evidence, and an exact whole-bundle payload inventory. Run
`scripts/verify-release-bundle.sh OUTPUT` before every use; pass
`OUTPUT/oci-context` to the existing OCI build wrapper.
