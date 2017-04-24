namespace fsh_net

[<AutoOpen>]
module Training =

    let GetModelGradients model input_batch target_batch wd_coefficient =
        GetZeroModel 7


    let GetLoss model input_batch target_batch wd_coefficient =
        0.0


    let GetClassificationPerformance model input_batch target_batch =
        
        let network_state = RunNetwork model input_batch
        let choices = network_state.ClassifierChoices
        let targetChoices = IntChoices(target_batch)
        let mutable sum = 0.0
        for i = 0 to choices.Length - 1 do
            if choices.[i] <> targetChoices.[i]
                then sum <- sum + 1.0

        sum / (double choices.Length)


    let TrainFromBatch (model: Model) input_batch target_batch (momentum: Model) (parameters: NetworkParameters) =
    
        let gradient = GetModelGradients model input_batch target_batch parameters.WeightDecayCoefficient
            
        momentum.Scale parameters.MomentumCoefficient
        momentum.Subtract gradient

        let update = momentum
        update.Scale parameters.LearningRate
        model.Add update