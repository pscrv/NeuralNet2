namespace fsh_net

open MathNet.Numerics.LinearAlgebra
open System

[<AutoOpen>]
module NetRun =

    let logistic (input: double) = (double) 1.0 / (1.0 + Math.Exp -input)

  

    let IntChoices (matrix: Matrix<double>) =
        let choices : int array = Array.zeroCreate matrix.ColumnCount
        for index, column in matrix.EnumerateColumnsIndexed() do
            choices.[index] <- column.MaximumIndex()
        choices



    let RunNetwork (network_data: NetworkData) (batch: TrainingData) =

        let hidden_input = network_data.Model.InputToHidden.Multiply batch.Input
        let hidden_output = hidden_input.Map (fun x -> logistic x)
        let class_input = network_data.Model.HiddenToClassifier.Multiply hidden_output
        let class_choices = IntChoices (class_input)
         
        {
            Input = batch.Input;
            HiddenInput = hidden_input;
            HiddenOutput = hidden_output;
            ClassifierInput = class_input;
            ClassifierChoices = class_choices;
        }

