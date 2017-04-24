namespace FNet

open System

module Main = 

    [<EntryPoint>]
    let main argv = 

        let parameters =  
            {
                NumberOfInputs = 256;
                NumberOfHiddenUnits = 37;
                NumberOfClasses = 10;
                NumberOfIterations = 1000;
                BatchSize = 375;
                LearningRate = 0.5;
                MomentumCoefficient = 0.9;
                WeightDecayCoefficient = 0.01
            }            

        let trainingState = 
            {
                Model = InitialiseModel parameters;
                Momentum = GetZeroModel parameters;
                Gradient = GetZeroModel parameters;
                LossRecords = Losses()
            }
        
        let data = DataExtraction.ReadMatFile      

        let result = TrainOnBatches parameters trainingState data 

        
        Console.ReadLine() |> ignore
        0
