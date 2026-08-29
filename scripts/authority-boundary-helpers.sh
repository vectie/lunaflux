#!/bin/sh

# Shared by the independently runnable authority-surface validators. The caller
# owns the aggregate "failed" flag and has already entered the repository root.

assert_self_factory_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg \
    "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
    "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public Self factory allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_escape_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$({
    rg "^pub fn ${authority_type}::.* -> Self( raise .*)?$" \
      "$interface" || true
    rg --pcre2 \
      "^pub fn .* -> .*(?<![A-Za-z0-9_])${authority_type}(?![A-Za-z0-9_]).*$" \
      "$interface" || true
  } | sed 's/^pub fn //; s/(.*$//' | sort -u)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public authority escape allowlist drifted" >&2
    failed=1
  fi
}

assert_authority_method_allowlist() {
  interface=$1
  authority_type=$2
  expected=$3
  actual="$(rg "^pub fn ${authority_type}::" "$interface" | sed \
    "s/^pub fn ${authority_type}:://; s/(.*$//" | sort || true)"
  if [ "$actual" != "$expected" ]; then
    printf '%s\n' \
      "${authority_type} public method allowlist drifted" >&2
    failed=1
  fi
}
