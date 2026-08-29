#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

fail_matches() {
  description=$1
  shift
  if matches=$(rg -n "$@" 2>/dev/null); then
    printf '%s\n%s\n' "$description" "$matches" >&2
    failed=1
  fi
}

# The bounded Luna tokenizer worker is synchronous cooperative work. It owns
# only fixed scratch and resumable scalar state; async orchestration, sockets,
# process authority, and dynamically growing work storage stay above it.
if [ -d tokenizer ]; then
  fail_matches \
    'Luna tokenizer work acquired async, transport, process, or native authority:' \
    --glob 'tokenizer/luna_worker*.mbt' \
    --glob 'tokenizer/luna_input_write.mbt' --glob 'tokenizer/moon.pkg' \
    'moonbitlang/async|socket|internal/process|worker_process|extern\s+"[cC]"|#external|pub async fn'
  fail_matches \
    'Luna tokenizer warmed work uses dynamically growing Array or Map storage:' \
    --pcre2 --glob 'tokenizer/luna_worker*.mbt' \
    --glob 'tokenizer/luna_input_write.mbt' \
    '(^|[^A-Za-z])((Array|Map)\[|Array::|Map\()'
  if ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerInputWrite \{\n  priv worker : LunaTokenizerWorker\n  priv epoch : UInt64\n\}' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerWork \{\n  priv worker : LunaTokenizerWorker\n  priv epoch : UInt64\n\}' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q --pcre2 -U \
      '#valtype\npub struct LunaTokenizerStepBudget \{\n  priv work_units : Int\n\}' \
      tokenizer/luna_worker_types.mbt ||
    rg -n --pcre2 -U \
      'pub struct LunaTokenizer(?:InputWrite|Work|Worker|StepBudget)(?s:.*?)derive\([^)]*Debug' \
      tokenizer/luna_worker*.mbt tokenizer/luna_input_write.mbt; then
    printf '%s\n' \
      'Luna tokenizer work, worker, and budget must remain opaque without Debug' >&2
    failed=1
  fi
  worker_api="$(rg --no-filename -o '^pub fn LunaTokenizerWorker::[a-z_]+' \
    tokenizer/luna_worker*.mbt tokenizer/luna_input_write.mbt | sort)"
  if [ "$worker_api" != \
    $'pub fn LunaTokenizerWorker::begin_bytes\npub fn LunaTokenizerWorker::begin_luna_input\npub fn LunaTokenizerWorker::new\npub fn LunaTokenizerWorker::required_byte_cells\npub fn LunaTokenizerWorker::required_int_cells' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer worker must expose only construction and epoch-bound begin:' \
      "$worker_api" >&2
    failed=1
  fi
  input_write_api="$(rg --no-filename -o '^pub fn LunaTokenizerInputWrite::[a-z_]+' \
    tokenizer/luna_input_write.mbt | sort)"
  if [ "$input_write_api" != \
    $'pub fn LunaTokenizerInputWrite::abort\npub fn LunaTokenizerInputWrite::finish\npub fn LunaTokenizerInputWrite::push_byte' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer input-write capability surface drifted:' \
      "$input_write_api" >&2
    failed=1
  fi
  work_api="$(rg --no-filename -o '^pub fn LunaTokenizerWork::[a-z_]+' \
    tokenizer/luna_worker*.mbt | sort)"
  if [ "$work_api" != \
    $'pub fn LunaTokenizerWork::abort\npub fn LunaTokenizerWork::copy_tokens_to\npub fn LunaTokenizerWork::last_work_units\npub fn LunaTokenizerWork::progress\npub fn LunaTokenizerWork::token_count\npub fn LunaTokenizerWork::token_status\npub fn LunaTokenizerWork::was_truncated' ]; then
    printf '%s\n%s\n' \
      'Luna tokenizer work capability surface drifted:' "$work_api" >&2
    failed=1
  fi
  if [ -f tokenizer/pkg.generated.mbti ]; then
    if [ "$(rg -c '^pub fn LunaTokenizerWorker::' \
        tokenizer/pkg.generated.mbti)" != '5' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerInputWrite::' \
        tokenizer/pkg.generated.mbti)" != '3' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerWork::' \
        tokenizer/pkg.generated.mbti)" != '7' ] ||
      [ "$(rg -c '^pub fn LunaTokenizerStepBudget::' \
        tokenizer/pkg.generated.mbti)" != '2' ] ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerStepBudget \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerInputWrite \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWork \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q --pcre2 -U \
        'pub struct LunaTokenizerWorker \{\n  // private fields\n\}' \
        tokenizer/pkg.generated.mbti ||
      rg -q --pcre2 \
        '^pub struct LunaTokenizer(?:InputWrite|StepBudget|Work|Worker)\(|^pub fn LunaTokenizer(?:InputWrite|StepBudget|Work|Worker)::[^\n]*(?:owner|epoch|storage)|LunaTokenizer(?:InputWrite|StepBudget|Work|Worker).*derive\([^)]*Debug' \
        tokenizer/pkg.generated.mbti; then
      printf '%s\n' \
        'generated Luna tokenizer authorities must remain exact and opaque' >&2
      failed=1
    fi
    if ! rg -q --pcre2 -U \
        'pub\(all\) enum LunaTokenizerProgress \{\n  LunaTokenizerPending\n  LunaTokenizerReady\n\}' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::begin_bytes\(Self, Bytes, special_policy~ : SpecialTokenPolicy, output_limit~ : Int, truncation~ : TruncationPolicy\) -> LunaTokenizerWork raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::begin_luna_input\(Self, byte_count~ : Int, special_policy~ : SpecialTokenPolicy, output_limit~ : Int, truncation~ : TruncationPolicy\) -> LunaTokenizerInputWrite raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::required_byte_cells\(Int\) -> UInt64 raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerWorker::new\(TokenizerSpec, input_capacity~ : Int, output_capacity~ : Int, LunaTokenizerStepBudget\) -> Self raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::abort\(Self\) -> Unit raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::finish\(Self\) -> LunaTokenizerWork raise TokenizerError$' \
        tokenizer/pkg.generated.mbti ||
      ! rg -q '^pub fn LunaTokenizerInputWrite::push_byte\(Self, Byte\) -> Unit raise TokenizerError$' \
        tokenizer/pkg.generated.mbti; then
      printf '%s\n' \
        'generated Luna tokenizer lifecycle surface drifted' >&2
      failed=1
    fi
  fi
  if ! rg -q 'priv input : FixedArray\[Byte\]' \
      tokenizer/luna_worker_types.mbt ||
    rg -n 'priv (mut )?input : Bytes' tokenizer/luna_worker_types.mbt ||
    rg -n '^pub fn LunaTokenizer(?:Worker|InputWrite)::(?:input|source|raw_bytes|request_bytes)' \
      tokenizer/pkg.generated.mbti ||
    ! rg -q 'self\.input_offset .* self\.input_length' \
      tokenizer/luna_worker_progress.mbt; then
    printf '%s\n' \
      'Luna tokenizer worker no longer proves fixed owned byte input' >&2
    failed=1
  fi
  if ! rg -q 'const LUNA_TOKENIZER_MAX_STEP_WORK_UNITS : Int = 65536' \
      tokenizer/luna_worker_types.mbt ||
    ! rg -q 'length > worker\.budget\.work_units()' tokenizer/luna_worker.mbt ||
    ! rg -q --pcre2 -U \
      'pub fn LunaTokenizerWork::progress(?s).*let maximum = worker\.budget\.work_units\(\).*while used < maximum' \
      tokenizer/luna_worker_progress.mbt;
  then
    printf '%s\n' \
      'Luna tokenizer progress/copy work bounds are not enforced at the owner' >&2
    failed=1
  fi
fi

# Incremental output owns fixed per-request decode/matcher state only. Socket,
# scheduler, worker, filesystem, and native authority stay outside this leaf.
if [ -d service/incremental_output ]; then
  fail_matches \
    'incremental output imports outside inference contracts + tokenizer:' \
    --pcre2 --glob 'service/incremental_output/moon.pkg' \
    '"vectie/lunaflux/(?!contracts/inference"|tokenizer")'
  fail_matches \
    'incremental output must remain synchronous and native-ABI free:' \
    --glob 'service/incremental_output/*.mbt' \
    'pub async fn|extern\s+"[cC]"|#external'
  if [ -f service/incremental_output/pkg.generated.mbti ]; then
    if ! rg -q \
      '^pub fn IncrementalOutput::push_token_into\(Self, Int, FixedArray\[Byte\], destination_offset~ : Int\)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output must retain its fixed-destination token API' >&2
      failed=1
    fi
    if ! rg -q \
      '^pub fn LunaIncrementalOutputWorkspace::begin_luna_request_semantics\(Self, @inference\.LunaRequestSemanticView, special_policy~ : @tokenizer\.DecodeSpecialPolicy\) -> LunaIncrementalOutputWork raise IncrementalOutputError$' \
      service/incremental_output/pkg.generated.mbti ||
      ! rg -q \
        '^pub fn LunaIncrementalOutputWorkspace::required_reference_cells\(@inference\.InferenceLimits\) -> Int raise IncrementalOutputError$' \
        service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output semantic-view setup surface drifted' >&2
      failed=1
    fi
    if rg -n \
      '^pub fn LunaIncrementalOutput[^ ]*::.*@inference\.LunaRequestSemantic(Storage|Write|Work|Lease)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output acquired semantic storage or release authority' >&2
      failed=1
    fi
    if rg -n --pcre2 -U \
      'pub struct IncrementalOutput \{\n  (?!// private fields)' \
      service/incremental_output/pkg.generated.mbti; then
      printf '%s\n' \
        'incremental output state must remain opaque' >&2
      failed=1
    fi
  fi
fi


if [ "$failed" -ne 0 ]; then
  exit 1
fi
