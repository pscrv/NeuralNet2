namespace fsh_net

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module ModelFunctions =

    let numberOfInputs = 256
    let numberOfClasses = 10

    let InitialiseModel (numberOfHiddenUnits: int) = 
        let numberOfInputToHiddenWeights = numberOfInputs * numberOfHiddenUnits
        let _in_to_hid = Matrix.Cos(Matrix.Build.Dense(numberOfHiddenUnits, numberOfInputs, (fun row col -> (double) (numberOfInputs * row + col))))
        let _hid_to_class = Matrix.Cos(Matrix.Build.Dense(numberOfClasses, numberOfHiddenUnits, (fun row col -> (double) (numberOfClasses * row + col + numberOfInputToHiddenWeights))))
        
        new Model(_in_to_hid, _hid_to_class)

    
    let GetZeroModel (numberOfHiddenUnits: int) = 
        let numberOfInputToHiddenWeights = numberOfInputs * numberOfHiddenUnits
        let _in_to_hid = Matrix.Build.Dense(numberOfHiddenUnits, numberOfInputs, (fun row col -> 0.0))
        let _hid_to_class =Matrix.Build.Dense(numberOfClasses, numberOfHiddenUnits, (fun row col -> 0.0))
        
        new Model(_in_to_hid, _hid_to_class)
