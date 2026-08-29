///|
const MAX_PROFILES : Int = 256

///|
const MAX_CONTRACTS : Int = 65536

///|
const MAX_OPERANDS : Int = 32

///|
pub struct TensorParallelLaunchContractLimits {
  priv max_profiles : Int
  priv max_contracts : Int
  priv max_operands_per_contract : Int
} derive(Debug, Eq)

///|
pub fn TensorParallelLaunchContractLimits::new(
  max_profiles~ : Int,
  max_contracts~ : Int,
  max_operands_per_contract~ : Int,
) -> TensorParallelLaunchContractLimits raise TensorParallelLaunchContractError {
  if max_profiles <= 0 ||
    max_profiles > MAX_PROFILES ||
    max_contracts <= 0 ||
    max_contracts > MAX_CONTRACTS ||
    max_operands_per_contract <= 0 ||
    max_operands_per_contract > MAX_OPERANDS {
    raise invalid(InvalidLimit)
  }
  { max_profiles, max_contracts, max_operands_per_contract }
}

///|
pub fn TensorParallelLaunchContractLimits::max_profiles(
  self : TensorParallelLaunchContractLimits,
) -> Int {
  self.max_profiles
}

///|
pub fn TensorParallelLaunchContractLimits::max_contracts(
  self : TensorParallelLaunchContractLimits,
) -> Int {
  self.max_contracts
}

///|
pub fn TensorParallelLaunchContractLimits::max_operands_per_contract(
  self : TensorParallelLaunchContractLimits,
) -> Int {
  self.max_operands_per_contract
}
