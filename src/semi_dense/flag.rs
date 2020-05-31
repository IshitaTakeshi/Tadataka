
#[derive(Debug)]
#[derive(PartialEq)]
pub enum Flag {
    Success,
    HypothesisOutOfSerchRange,
    KeyOutOfRange,
    RefCloseOutOfRange,
    RefFarOutOfRange,
    RefEpipolarTooShort,
    InsufficientGradient,
    NegativePriorDepth,
    NegativeRefDepth,
    NotProcessed
}
