namespace fsh_net

open System
open System.IO

open Accord.IO


module Main =

    [<EntryPoint>]
    let main argv = 

        let parameters =  
            {
                NumberOfHiddenUnits = 7;
                NumberOfIterations = 100;
                BatchSize = 4;
                LearningRate = 0.01;
                MomentumCoefficient = 0.0;
                WeightDecayCoefficient = 0.0
            }

        let data = DataExtraction.ReadMatFile
        let model = InitialiseModel parameters.NumberOfHiddenUnits
        let momentum = GetZeroModel parameters.NumberOfHiddenUnits
        let weight_decay_coefficient = 0

        let mutable training_losses = []
        let mutable validation_losses = []

        let trainingCaseCount = data.Training.Target.ColumnCount
        for iteration_count = 1 to parameters.NumberOfIterations do

            let input_batch = data.Training.Input.SubMatrix (0, data.Training.Input.RowCount, iteration_count * parameters.BatchSize, parameters.BatchSize )
            let target_batch = data.Training.Target.SubMatrix (0, data.Training.Target.RowCount, iteration_count * parameters.BatchSize, parameters.BatchSize )    

            TrainFromBatch model input_batch target_batch momentum parameters

            training_losses <- GetLoss model input_batch target_batch  weight_decay_coefficient :: training_losses 
            validation_losses <- GetLoss model data.Validation.Input data.Validation.Target  weight_decay_coefficient :: validation_losses 

            if iteration_count % 10 = 0
                then printf "After %A optimization iterations, training data loss is %A, and validation data loss is %A\n\n" iteration_count training_losses.Head validation_losses.Head




        for datatype in [data.Training; data.Validation; data.Test] do
            printf "\nThe loss on the %A data is %A\n" datatype.Name  (GetLoss model datatype.Input datatype.Target weight_decay_coefficient)

            if weight_decay_coefficient <> 0
                then printf "The classification loss (i.e. without weight decay) on the %A data is %A\n" datatype.Name (GetLoss model, datatype.Input, datatype.Target, 0);
                
            printf "The classification error rate on the %A data is %A\n" datatype.Name (GetClassificationPerformance model datatype.Input datatype.Target);


        
        Console.ReadLine() |> ignore
        0 // return an integer exit code
