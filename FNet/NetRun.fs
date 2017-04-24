namespace FNet

open MathNet.Numerics.LinearAlgebra
open System

[<AutoOpen>]
module NetRun =

    let _getChoices (matrix: Matrix<double>) =
        let choices = Array.zeroCreate matrix.ColumnCount
        for index, column in matrix.EnumerateColumnsIndexed() do
            choices.[index] <- column.MaximumIndex()
        choices


    let RunNetwork (model : Model) (batchinput : Matrix<double>) =        
        let hidden_input = 
            batchinput
            |> model.InputToHidden.Multiply
        let hidden_output =
            fun x -> Logistic x 
            |> hidden_input.Map 
        let class_input = 
            hidden_output 
            |> model.HiddenToClassifier.Multiply
        let class_choices = 
            class_input
            |> _getChoices 
         
        {
            Input = batchinput;
            HiddenInput = hidden_input;
            HiddenOutput = hidden_output;
            ClassifierInput = class_input;
            ClassifierChoices = class_choices;
        }

