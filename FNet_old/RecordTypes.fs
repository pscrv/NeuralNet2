namespace FNet

module RecordTypes =

    type Parameters = {        
        NumberOfHiddenUnits : int
        NumberOfIterations  : int
        BatchSize : int
        LearningRate : double
        MomentumCoefficient : double
        WeightDecayCoefficient : double
    }

