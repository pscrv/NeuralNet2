namespace fsh_net

open MathNet.Numerics.LinearAlgebra
open System

[<AutoOpen>]
module NetRun =

    let logistic (input: double) = (double) 1.0 / (1.0 - Math.Exp input)

    let softmax (input: Matrix<double>) =
        Matrix.Build.Dense(input.RowCount, input.ColumnCount)


    let IntChoices (matrix: Matrix<double>) =
        let choices : int array = Array.zeroCreate matrix.ColumnCount
        for index, column in matrix.EnumerateColumnsIndexed() do
            choices.[index] <- column.MaximumIndex()
        choices



    let RunNetwork (model: Model) (input_batch: Matrix<double>) =

        let hidden_input = model.InputToHidden.Multiply input_batch
        let hidden_output = hidden_input.Map (fun x -> logistic x)
        let class_input = model.HiddenToClassifier.Multiply hidden_output
        let class_choices = IntChoices (class_input)
        let softmax_output = softmax class_input

         
        {
            Input = input_batch;
            HiddenInput = hidden_input;
            HiddenOutput = hidden_output;
            ClassifierInput = class_input;
            ClassifierChoices = class_choices;
            SoftMaxOutput = softmax_output
        }

