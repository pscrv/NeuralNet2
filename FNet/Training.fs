namespace FNet

open MathNet.Numerics.LinearAlgebra


[<AutoOpen>]
module Training =

    let _getModelGradients (model : Model) networkState (target : Matrix<double>) =
        
        let hid_derivative = 
            LogisticDerivative_f_output 
            |> networkState.HiddenOutput.Map

        let subtract_target (matrix : Matrix<double>) =
            matrix.Subtract target

        let divide_by_column_count (matrix : Matrix<double>) =
            matrix.Divide (double matrix.ColumnCount)

        let multiply_by_transpose_of_network_input (matrix : Matrix<double>) =
            matrix.TransposeAndMultiply networkState.Input


        let class_input_gradient =
            networkState.ClassifierInput
                |> GetSoftMaxOutput
                |> subtract_target
                |> divide_by_column_count

        let hid_to_class_gradient =
            networkState.HiddenOutput
            |> class_input_gradient.TransposeAndMultiply

        let input_to_hid_gradient = 
            class_input_gradient
                |> model.HiddenToClassifier.TransposeThisAndMultiply
                |> hid_derivative.PointwiseMultiply
                |> multiply_by_transpose_of_network_input

        new Model(input_to_hid_gradient, hid_to_class_gradient)

    let _calculateLoss classifierInput (model : Model) (target : Matrix<double>) weight_decay_coefficient  =
        
        let log_softmax_output = 
            classifierInput
            |> GetSoftMaxLogOutput 

        let classification_loss = 
            - (log_softmax_output
            |> target.PointwiseMultiply
            |> ColumnSums
            |> VectorElementMean)

        let weight_decay_loss = 
            weight_decay_coefficient * model.SumOfSquares / 2.0

        classification_loss + weight_decay_loss
        

    let _getLoss network_state training_state (target : Matrix<double>) weight_decay_coefficient =
        _calculateLoss network_state.ClassifierInput training_state.Model target weight_decay_coefficient


    //TODO: refactor here
    let _getClassificationPerformance model input_batch target_batch =      
    
        let _intChoices (matrix: Matrix<double>) =
            let choices : int array = Array.zeroCreate matrix.ColumnCount
            for index, column in matrix.EnumerateColumnsIndexed() do
                choices.[index] <- column.MaximumIndex()
            choices
          
        let network_state = RunNetwork model input_batch
        let choices = network_state.ClassifierChoices
        let targetChoices = _intChoices(target_batch)

        let mutable sum = 0.0
        for i = 0 to choices.Length - 1 do
            if choices.[i] <> targetChoices.[i]
                then sum <- sum + 1.0

        sum / (double choices.Length)



    let _trainBatch training_state parameters batch =
        let networkState =
            batch.Input
            |> RunNetwork training_state.Model

        // TODO: refactor here
        use modelGradients = _getModelGradients training_state.Model networkState batch.Target

        if parameters.WeightDecayCoefficient <> 0.0 then
            use weight_decay_gradient = training_state.Model.Copy
            weight_decay_gradient.Scale parameters.WeightDecayCoefficient
            modelGradients.Add weight_decay_gradient

        training_state.Momentum.Scale parameters.MomentumCoefficient
        training_state.Momentum.Subtract modelGradients

        // TODO: refactor here
        use update = training_state.Momentum.Copy
        update.Scale parameters.LearningRate
        training_state.Model.Add update
            


    let _log_losses training_state data parameters = 
        let network_state_training = RunNetwork training_state.Model data.Training.Input
        let network_state_validation = RunNetwork training_state.Model data.Validation.Input
        (
        _getLoss network_state_training training_state data.Training.Target parameters.WeightDecayCoefficient,
        _getLoss network_state_validation training_state data.Validation.Target parameters.WeightDecayCoefficient)
        |> training_state.LossRecords.AddLossPair 


               
    let _report training_state data parameters =
        for datatype in [data.Training; data.Validation; data.Test] do
            let network_state = RunNetwork training_state.Model datatype.Input
            let loss_without_weight_decay = _getLoss network_state training_state datatype.Target 0.0
                        
            match parameters.WeightDecayCoefficient with
            | 0.0 -> printf "\nThe loss on the %A data is %A\n" datatype.Name loss_without_weight_decay
                     
            | _   -> _getLoss network_state training_state datatype.Target parameters.WeightDecayCoefficient
                     |> printf "\nThe loss on the %A data is %A\n" datatype.Name                   
                     printf "The classification loss (i.e. without weight decay) on the %A data is %A\n" datatype.Name loss_without_weight_decay

            _getClassificationPerformance training_state.Model datatype.Input datatype.Target
            |> printf "The classification error rate on the %A data is %A\n" datatype.Name 




    let TrainOnBatches parameters training_state data =  
    
        for iteration_count = 0 to parameters.NumberOfIterations do
            MakeBatch data.Training iteration_count parameters.BatchSize
                |> _trainBatch training_state parameters
            _log_losses training_state data parameters

            match iteration_count with
               | x when x % 10 = 0 -> 
                   printf "After %A iterations, training data loss is %A, and validation data loss is %A\n\n" 
                          iteration_count 
                          training_state.LossRecords.LatestTrainingLoss 
                          training_state.LossRecords.LatestValidationLoss
               | _ -> () 

        _report training_state data parameters


