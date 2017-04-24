namespace FNet

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module RecordTypes =

    type Parameters = {
        NumberOfInputs : int        
        NumberOfHiddenUnits : int
        NumberOfClasses : int
        NumberOfIterations  : int
        BatchSize : int
        LearningRate : double
        MomentumCoefficient : double
        WeightDecayCoefficient : double
    }

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