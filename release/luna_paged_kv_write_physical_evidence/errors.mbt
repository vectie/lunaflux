///|
pub(all) enum PagedKvWritePhysicalRule {
  Canonical
  Digest
  Policy
  Candidate
  Compiled
  Target
  Oracle
  Canary
  Sanitizer
  Resource
  Seal
  Authority
} derive(Debug, Eq)

///|
pub(all) suberror PagedKvWritePhysicalError {
  Invalid(PagedKvWritePhysicalRule)
  Mismatch(PagedKvWritePhysicalRule)
  LimitExceeded
} derive(Debug, Eq)
