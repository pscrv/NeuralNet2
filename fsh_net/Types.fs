namespace fsh_net

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module Types =

    type TrainingData = {
        Name: string
        Input: Matrix<double>
        Target: Matrix<double>
    }

    type Data = {
        Training: TrainingData
        Validation: TrainingData
        Test: TrainingData
    }

    type NetworkParameters = {        
        NumberOfHiddenUnits : int
        NumberOfIterations  : int
        BatchSize : int
        LearningRate : double
        MomentumCoefficient : double
        WeightDecayCoefficient : double
    }



