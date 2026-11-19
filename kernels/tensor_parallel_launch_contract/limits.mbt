///|
const MAX_PROFILES : Int32 = 256

///|
const MAX_CONTRACTS : Int32 = 65536

///|
const MAX_OPERANDS : Int32 = 32

///|
pub struct TensorParallelLaunchContractLimits {
  priv max_profiles : Int32
  priv max_contracts : Int32
  priv max_operands_per_contract : Int32
} derive(Debug, Eq)

///|
pub fn TensorParallelLaunchContractLimits::new(
  max_profiles~ : Int32,
  max_contracts~ : Int32,
  max_operands_per_contract~ : Int32,
) -> TensorParallelLaunchContractLimits raise TensorParallelLaunchContractError {
  if max_profiles <= 0 ||
    max_profiles > MAX_PROFILES ||
    max_contracts <= 0 ||
    max_contracts > MAX_CONTRACTS ||
    max_operands_per_contract <= 0 ||
    max_operands_per_contract > MAX_OPERANDS {
    raise invalid(InvalidLimit)
  }
  { max_profiles, max_contracts, max_operands_per_contract, }
}

///|
pub fn TensorParallelLaunchContractLimits::max_profiles(
  self : TensorParallelLaunchContractLimits,
) -> Int32 {
  self.max_profiles
}

///|
pub fn TensorParallelLaunchContractLimits::max_contracts(
  self : TensorParallelLaunchContractLimits,
) -> Int32 {
  self.max_contracts
}

///|
pub fn TensorParallelLaunchContractLimits::max_operands_per_contract(
  self : TensorParallelLaunchContractLimits,
) -> Int32 {
  self.max_operands_per_contract
}
