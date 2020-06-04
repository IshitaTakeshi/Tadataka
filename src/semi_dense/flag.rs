
#[derive(Debug)]
#[derive(PartialEq)]
pub enum Flag {
    Success = 0,
    HypothesisOutOfSerchRange = -1,
    KeyOutOfRange = -2,
    RefCloseOutOfRange = -3,
    RefFarOutOfRange = -4,
    RefEpipolarTooShort = -5,
    InsufficientGradient = -6,
    NegativePriorDepth = -7,
    NegativeRefDepth = -8,
    NotProcessed = -9
}
