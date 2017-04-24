namespace FNet

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module SoftMax =

    /// For each input vector element, we compute exp(x[i]) / Sum_i(exp(x[i]))
    /// but to avoide big numbers, we subtract the maximum imput from each imput, so all
    /// expontents are negative.
    /// There is an alternative (below) but I do not (yet) see the advantage
    let GetSoftMaxOutput (classifierInput : Matrix<double>) =
        classifierInput
        |> ColumnMaxima 
        |> (SubtractVectorFromEachRow classifierInput) 
        |> Matrix.Exp 
        |> NormaliseColumns   
        

    /// Alternative:
    /// Calculate log(sum_i(exp(x[i] - max))) + max; call this y
    /// Calculate exp(x[i] - y)
    /// We get the log of the output as a partial result
    /// But overall, I see no fewer operatations and no advantage in terms of exponents


    let GetSoftMaxLogOutput (classifierInput : Matrix<double>) =
        classifierInput
        |> GetSoftMaxOutput
        |> Matrix.Log


