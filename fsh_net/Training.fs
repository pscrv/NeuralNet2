namespace fsh_net

open MathNet.Numerics.LinearAlgebra


[<AutoOpen>]
module Training =

    let MakeNormaliser (matrix: Matrix<double>) =
        let exp_matrix =
            matrix 
                |> ColumnMaxima
                |> (SubtractVectorFromEachRow matrix)
                |> Matrix.Exp
        let log_summed_exps = 
            exp_matrix.ColumnSums() 
                |> Vector.Log
        log_summed_exps.Add(matrix |> ColumnMaxima)

    let GetLogProbabilities network_state = 
            network_state.ClassifierInput
                |> MakeNormaliser
                |> (SubtractVectorFromEachRow network_state.ClassifierInput)


    let GetModelGradients (network_data: NetworkData) (network_state: NetworkState) (target: Matrix<double>) parameters =

        let hid_derivative = network_state.HiddenOutput.PointwiseMultiply(network_state.HiddenOutput.Map(fun x -> 1.0 - x))

        let class_input_gradient =
            network_state
                |> GetLogProbabilities
                |> Matrix.Exp
                |> fun x -> x.Subtract target
                |> fun x -> x.Divide (double target.ColumnCount)

        let hid_to_class_gradient = class_input_gradient.Multiply(network_state.HiddenOutput.Transpose())

        let input_to_hid_gradient = 
            class_input_gradient
                |> network_data.Model.HiddenToClassifier.TransposeThisAndMultiply
                |> hid_derivative.PointwiseMultiply
                |> network_state.Input.TransposeAndMultiply
                |> fun x -> x.Transpose()

        // Should have weight-decay gradient here too
        // octave code: model.input_to_hid * wd_coefficient  and the same for hid_to_class
        new Model(input_to_hid_gradient, hid_to_class_gradient)
           




    let GetLoss network_data weight_decay_coefficient network_state (target: Matrix<double>) =

        let classification_loss = 
            network_state
                |> GetLogProbabilities
                |> target.PointwiseMultiply
                |> SumElements 
                |> fun x -> x / (double target.ColumnCount)

        let weight_decay_loss = weight_decay_coefficient * network_data.Model.SumOfSquares / 2.0
        classification_loss + weight_decay_loss

        



    let GetClassificationPerformance network_data input_batch target_batch =        
        let network_state = RunNetwork network_data input_batch
        let choices = network_state.ClassifierChoices
        let targetChoices = IntChoices(target_batch)
        let mutable sum = 0.0
        for i = 0 to choices.Length - 1 do
            if choices.[i] <> targetChoices.[i]
                then sum <- sum + 1.0

        sum / (double choices.Length)


    let UpdateNetwork (network_data: NetworkData) (network_state: NetworkState) (target: Matrix<double>) (parameters: NetworkParameters) =

            GetModelGradients network_data network_state target parameters
                |> network_data.Momentum.ScaleThenSubtract parameters.MomentumCoefficient

            use update = network_data.Momentum.Copy 
            parameters.LearningRate
                |> update.Scale 
            update
                |> network_data.Model.Add

        

    let TrainOnBatches (network_data: NetworkData) (data: Data) (parameters: NetworkParameters) =    
        for iteration_count = 1 to parameters.NumberOfIterations do
                
            let batch =  MakeBatch data.Training iteration_count parameters.BatchSize
            let network_state = RunNetwork network_data batch
            UpdateNetwork network_data network_state batch.Target parameters


            let network_state = RunNetwork network_data data.Training
            network_data.LossRecords.AddTrainingLoss <| GetLoss network_data parameters.WeightDecayCoefficient network_state data.Training.Target
            let network_state = RunNetwork network_data data.Validation
            network_data.LossRecords.AddValidationLoss <| GetLoss network_data parameters.WeightDecayCoefficient network_state data.Validation.Target

            match iteration_count with
               | x when x % 10 = 0 -> 
                   printf "After %A optimization iterations, training data loss is %A, and validation data loss is %A\n\n" 
                          iteration_count 
                          network_data.LossRecords.LatestTrainingLoss 
                          network_data.LossRecords.LatestValidationLoss
               | _ -> ()  

               
        for datatype in [data.Training; data.Validation; data.Test] do
            let network_state = RunNetwork network_data datatype
            let loss = GetLoss network_data parameters.WeightDecayCoefficient network_state datatype.Target
            printf "\nThe loss on the %A data is %A\n" datatype.Name loss

            if parameters.WeightDecayCoefficient <> 0.0
                then 
                let network_state = RunNetwork network_data datatype
                let loss = GetLoss network_data 0.0 network_state datatype.Target
                printf "The classification loss (i.e. without weight decay) on the %A data is %A\n" datatype.Name loss
                
            printf "The classification error rate on the %A data is %A\n" datatype.Name (GetClassificationPerformance network_data datatype datatype.Target);

