namespace FNet

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module ModelFunctions =

    let InitialiseModel (parameters) = 

        let _elementCount rowCount offset =
            fun (i: int) (j: int) -> double (i + (j * rowCount) + offset)

        let _in_to_hid =
            (parameters.NumberOfHiddenUnits, 
                parameters.NumberOfInputs, 
                _elementCount parameters.NumberOfHiddenUnits 0)
            |> Matrix.Build.Dense
            |> Matrix.Cos 

        let _hid_to_class =         
            (parameters.NumberOfClasses, 
                parameters.NumberOfHiddenUnits,
                _elementCount parameters.NumberOfClasses (parameters.NumberOfInputs * parameters.NumberOfHiddenUnits))
            |> Matrix.Build.Dense
            |> Matrix.Cos
        
        new Model(_in_to_hid, _hid_to_class)

    
    let GetZeroModel (parameters) = 
        
        let _in_to_hid = 
            (parameters.NumberOfHiddenUnits, parameters.NumberOfInputs)
            |> Matrix.Build.Dense

        let _hid_to_class =
            (parameters.NumberOfClasses, parameters.NumberOfHiddenUnits) 
            |> Matrix.Build.Dense
        
        new Model(_in_to_hid, _hid_to_class)
