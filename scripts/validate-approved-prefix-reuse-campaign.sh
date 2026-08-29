#!/usr/bin/env bash
set -eu

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

request='tests/approved_model_spawned_physical/prefix_request.mbt'
campaign='tests/approved_model_spawned_physical/prefix_campaign.mbt'
tests='tests/approved_model_spawned_physical/prefix_campaign_wbtest.mbt'
referee='tests/reference_corpus/prefix_referee.json'
referee_test='tests/reference_corpus/reference_corpus_test.mbt'
materializer='scripts/materialize-approved-tiny-bf16-launch.sh'

[ "$(sha256_file "$referee")" = \
  765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3 ] ||
  fail 'approved prefix referee digest changed'
rg -Fq \
  '`prefix_referee.json` | 792 | `765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3`' \
  tests/reference_corpus/MANIFEST.md ||
  fail 'approved prefix referee provenance is not documented'

for anchor in \
  'PREFIX_REFEREE_FIRST_TOKEN : Int = 1355' \
  'PREFIX_REFEREE_SECOND_TOKEN : Int = 1240' \
  '[1, 229, 153, 132, 75, 104, 111, 111]' \
  '[1, 229, 153, 132, 75, 104, 111, 111, 114]' \
  'PREFIX_CAMPAIGN_FIRST_REQUEST_ID : UInt64 = 101UL' \
  'PREFIX_CAMPAIGN_SECOND_REQUEST_ID : UInt64 = 102UL' \
  'b"spawned-prefix-proof"' \
  'ReadWrite' \
  'ReadOnly' \
  'max_input_tokens=9' \
  'first_cached_input_tokens=0' \
  'second_cached_input_tokens=8'; do
  rg -Fq "$anchor" "$request" "$campaign" ||
    fail "approved prefix campaign anchor lost: $anchor"
done

for anchor in \
  'approved_scalar_bf16_reference_executor' \
  'prefix campaign referee is digest pinned and independently replayable' \
  '@reference.execute(' \
  'assert_eq(execution.selection().token_id(), expected_tokens[index])'; do
  rg -Fq "$anchor" "$referee" "$referee_test" ||
    fail "independent BF16 referee replay anchor lost: $anchor"
done

for anchor in \
  '[_, "prefix-reuse", deployment_argument, expected_worker_sha256]' \
  'run_prefix_campaign(deployment_argument, expected_worker_sha256)'; do
  rg -Fq "$anchor" tests/approved_model_spawned_physical/main.mbt ||
    fail "prefix campaign dispatch lost: $anchor"
done

for anchor in \
  'prefix-reuse-v1)' \
  '[ "$#" -eq 9 ] || usage' \
  'descriptor_max_input_tokens=9' \
  'tokenizer_max_output_tokens=9' \
  'prefix_enabled=true' \
  'max_prefix_entries=1' \
  'max_prefix_nodes=8' \
  'max_prefix_tokens_per_entry=8' \
  'max_prefix_pages=1' \
  'max_prefix_scope_bytes=32'; do
  rg -Fq "$anchor" "$materializer" ||
    fail "prefix launch profile lost: $anchor"
done

for anchor in \
  'owner.require_spawned_physical_handoff()' \
  'owner.validate_spawned_prefix_reuse(contract)' \
  'owner.phase() != Closed' \
  '!owner.cleanup_complete()' \
  'prefix_lookups=' \
  'prefix_hits=' \
  'prefix_publications=' \
  'prefix_tokens_reused=' \
  'kv_pages_free_initial=' \
  'kv_pages_free_before_close='; do
  rg -Fq "$anchor" "$campaign" ||
    fail "owner-retained prefix evidence lost: $anchor"
done

for hostile in \
  'prefix campaign refuses disabled readonly seed and cross-scope aliases' \
  'assert_true(hostile != canonical)' \
  'reader.cache().scope().as_string() != wrong_scope.as_string()' \
  'prefix expected-token substitutions cannot rewrite either request'; do
  rg -Fq "$hostile" "$tests" ||
    fail "prefix campaign hostile evidence lost: $hostile"
done

if rg -n \
  '@cuda\.|@device_worker|@worker_service|@worker_process|@online_tcp|nvrtc|PTX|JIT|Python' \
  "$request" "$campaign" "$tests"; then
  fail 'prefix campaign bypasses the opaque runtime owner'
fi

while IFS= read -r file; do
  [ "$(wc -l <"$file" | tr -d ' ')" -lt 500 ] ||
    fail "prefix campaign source exceeds file budget: $file"
done < <(printf '%s\n' "$request" "$campaign" "$tests")

moon info --target native >/dev/null
moon check --target native --deny-warn tests/approved_model_spawned_physical
moon test --target native --deny-warn tests/approved_model_spawned_physical
moon test --target native --deny-warn tests/reference_corpus

printf '%s\n' 'approved BF16 prefix-reuse campaign gate passed'
