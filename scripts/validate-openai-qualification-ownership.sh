#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
owner="$root/ops/runtime_instance/openai_qualification_owner.mbt"
binding="$root/ops/runtime_instance/openai_owner_binding.mbt"
production="$root/ops/runtime_instance/openai_production_owner.mbt"
production_types="$root/ops/runtime_instance/openai_production_types.mbt"
types="$root/ops/runtime_instance/types.mbt"
admission="$root/runtime/instance_admission/external_protocol.mbt"
admission_test="$root/runtime/instance_admission/external_protocol_wbtest.mbt"
opaque_policy="$root/ops/runtime_instance/opaque_openai_policy.mbt"

test "$(rg -c 'prepare_luna_online_openai_server' "$binding")" -eq 1
test "$(rg -c 'bind_luna_online_openai_server' "$binding")" -eq 1
rg -q 'OpenAiQualificationBinding => false' "$binding"
rg -q 'OpenAiEmbeddingProductionBinding' "$binding"
rg -q 'OpenAiOpaqueCliProductionBinding' "$binding"
rg -q 'OpenAiQualificationBinding' "$owner"
rg -q 'bind_openai_production' "$production"
rg -Fq 'self.readiness() != RuntimeReady' "$production"
rg -q 'RuntimeOpenAiProductionPolicy' "$production_types"
if rg -q 'RuntimeOpenAiProductionPolicy' "$owner"; then
  echo 'qualification policy must not acquire production binding authority' >&2
  exit 1
fi
rg -q 'if phase == Ready && listener_bound' \
  "$root/ops/runtime_instance/owner_status.mbt"
rg -q 'OpenAiQualificationReady' "$types"
rg -q '^fn external_protocol_admitted' "$admission"
rg -Fq 'OpenAiResponsesV1 => openai_service_available' "$admission"
rg -Fq 'test "OpenAI loopback intent requires the local service policy"' \
  "$admission_test"
rg -Fq 'test "OpenAI exposure fails closed for non-loopback and unapproved TLS"' \
  "$admission_test"
rg -Fq 'if credential is Some(inherited)' "$opaque_policy"
rg -Fq 'guard credential is Some(inherited) else {' "$opaque_policy"
rg -Fq 'guard instance.openai_service_policy() is Some(claims) else {' \
  "$opaque_policy"
rg -Fq 'inherited.take_for_policy(claims.maximum_credential_bytes())' \
  "$opaque_policy"
test "$(rg -Fc 'raise fail(InheritedCredentialBinding)' "$opaque_policy")" -ge 3
if rg -q \
  'ExternalProtocolAdmissionReport|ExternalProtocolAdmissionBlocker|inspect_external_protocol' \
  "$root/runtime/instance_admission"; then
  echo 'external protocol admission must remain a private fail-closed join' >&2
  exit 1
fi
rg -q 'RuntimeOpaqueOpenAiBinding' "$root/ops/runtime_instance/types.mbt"
rg -q 'prepare_opaque_cli' "$root/ops/runtime_instance/prepare.mbt"

if rg -n '@env|async/fs|getenv|read_file' \
  "$root/ops/runtime_instance/openai_qualification"*.mbt \
  "$binding" "$production" "$production_types" "$opaque_policy"; then
  echo 'OpenAI owners must not discover ambient credentials' >&2
  exit 1
fi

echo 'openai qualification ownership gate: pass'
