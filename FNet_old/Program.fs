﻿namespace FNet

open System

module Main = 

    [<EntryPoint>]
    let main argv = 

        let parameters =  
            {
                NumberOfHiddenUnits = 37;
                NumberOfIterations = 1000;
                BatchSize = 100;
                LearningRate = 0.01;
                MomentumCoefficient = 0.9;
                WeightDecayCoefficient = 0.0001
            }            

//        let trainingState = 
//            {
//                Model = InitialiseModel parameters.NumberOfHiddenUnits;
//                Momentum = GetZeroModel parameters.NumberOfHiddenUnits;
//                Gradient = GetZeroModel parameters.NumberOfHiddenUnits;
//                LossRecords = Losses()
//            }
//            
//        let data = DataExtraction.ReadMatFile      
//
//        TrainOnBatches networkdata data parameters
        
        Console.ReadLine() |> ignore
        0