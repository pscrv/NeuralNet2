namespace FNet

open System
open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module MutableTypes =

    type Losses () = 
        let mutable _trainingLosses = []
        let mutable _validationLosses = []

        member this.TrainingLosses
            with get() = _trainingLosses

        member this.ValidationLosses
            with get() = _validationLosses

        member this.LatestTrainingLoss
            with get() = _trainingLosses.Head

        member this.LatestValidationLoss
            with get() = _validationLosses.Head

        member this.AddLossPair (training_loss: double, validation_loss: double) =
            _trainingLosses <- training_loss :: _trainingLosses
            _validationLosses <- validation_loss :: _validationLosses

    

    type Model (in_hid: Matrix<double>, hid_class: Matrix<double>) =

        // TODO: check compatibility

        let mutable _input_to_hidden_weights = Matrix.Build.DenseOfMatrix in_hid
        let mutable _hidden_to_classifier_weights = Matrix.Build.DenseOfMatrix hid_class



        interface IDisposable with
            member this.Dispose() =
                _input_to_hidden_weights <- null
                _hidden_to_classifier_weights <- null


        member this.InputToHidden
            with get() = _input_to_hidden_weights

        member this.HiddenToClassifier
            with get() = _hidden_to_classifier_weights


        member this.Scale (scalar: double) = 
            _input_to_hidden_weights <- _input_to_hidden_weights.Multiply(scalar)
            _hidden_to_classifier_weights <- _hidden_to_classifier_weights.Multiply(scalar)

        member this.Subtract (other: Model) =
            _input_to_hidden_weights <- _input_to_hidden_weights.Subtract other.InputToHidden
            _hidden_to_classifier_weights <- _hidden_to_classifier_weights.Subtract other.HiddenToClassifier

        member this.ScaleThenSubtract (scalar: double) (other: Model) =
            this.Scale scalar
            this.Subtract other

        member this.Add (other: Model) =
            _input_to_hidden_weights <- _input_to_hidden_weights.Add other.InputToHidden
            _hidden_to_classifier_weights <- _hidden_to_classifier_weights.Add other.HiddenToClassifier

        member this.SumOfSquares =
            let input_to_hidden_squares = _input_to_hidden_weights.Map(fun x -> x * x)
            let hidden_to_classifier_squares = _hidden_to_classifier_weights.Map(fun x -> x * x)
            input_to_hidden_squares.RowSums().Sum() + hidden_to_classifier_squares.RowSums().Sum()

        member this.Copy =
            new Model(_input_to_hidden_weights, _hidden_to_classifier_weights)

     
            
    type NetworkState = {
        Input: Matrix<double>
        HiddenInput: Matrix<double>
        HiddenOutput: Matrix<double>
        ClassifierInput: Matrix<double>
        ClassifierChoices: int array
    }

            
    type TrainingState = {
        Model: Model
        Momentum: Model    
        Gradient: Model
        LossRecords: Losses
    }
            



