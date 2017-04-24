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

    type NetworkState = {
        Input: Matrix<double>
        HiddenInput: Matrix<double>
        HiddenOutput: Matrix<double>
        ClassifierInput: Matrix<double>
        ClassifierChoices: int array
        SoftMaxOutput: Matrix<double>
    }


    type Model (in_hid: Matrix<double>, hid_class: Matrix<double>) =
        let mutable _input_to_hidden_weights = in_hid
        let mutable _hidden_to_classifier_weights = hid_class

        member this.InputToHidden
            with get() = _input_to_hidden_weights

        member this.HiddenToClassifier
            with get() = _hidden_to_classifier_weights



        member this.Scale (scalar: double) = 
            _input_to_hidden_weights <- _input_to_hidden_weights.Multiply(scalar)

        member this.Subtract (other: Model) =
            _input_to_hidden_weights <- _input_to_hidden_weights.Subtract other.InputToHidden
            _hidden_to_classifier_weights <- _hidden_to_classifier_weights.Subtract other.HiddenToClassifier

        member this.Add (other: Model) =
            _input_to_hidden_weights <- _input_to_hidden_weights.Add other.InputToHidden
            _hidden_to_classifier_weights <- _hidden_to_classifier_weights.Add other.HiddenToClassifier

            


