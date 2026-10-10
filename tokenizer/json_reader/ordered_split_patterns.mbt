///|
const ORDERED_NUMBER_REGEX : String = "\\p{N}{1,3}"

///|
const ORDERED_SCRIPT_REGEX : String = "[一-龥぀-ゟ゠-ヿ]+"

///|
const ORDERED_WORD_REGEX : String = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
