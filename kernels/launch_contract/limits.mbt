///|
pub struct LaunchContractLimits {
  max_profiles : Int
  max_contracts : Int
  max_operands_per_contract : Int
} derive(Debug, Eq)

///|
pub fn LaunchContractLimits::new(
  max_profiles~ : Int,
  max_contracts~ : Int,
  max_operands_per_contract~ : Int,
) -> LaunchContractLimits raise LaunchContractError {
  if max_profiles <= 0 || max_profiles > 256 {
    raise InvalidLimit(ProfileCount)
  }
  if max_contracts <= 0 || max_contracts > 65536 {
    raise InvalidLimit(ContractCount)
  }
  if max_operands_per_contract <= 0 || max_operands_per_contract > 32 {
    raise InvalidLimit(OperandsPerContract)
  }
  { max_profiles, max_contracts, max_operands_per_contract }
}

///|
pub fn LaunchContractLimits::max_profiles(self : LaunchContractLimits) -> Int {
  self.max_profiles
}

///|
pub fn LaunchContractLimits::max_contracts(self : LaunchContractLimits) -> Int {
  self.max_contracts
}

///|
pub fn LaunchContractLimits::max_operands_per_contract(
  self : LaunchContractLimits,
) -> Int {
  self.max_operands_per_contract
}
