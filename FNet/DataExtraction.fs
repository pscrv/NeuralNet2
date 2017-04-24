namespace FNet

open System.IO
open Accord.IO
open MathNet.Numerics.LinearAlgebra

module DataExtraction =

    let ExtractData (reader: MatReader) (sort: string) = 
        {
            Name = sort; 
            Input =
                reader.["data"].[sort].["inputs"].Value 
                :?> double[,] 
                |> Matrix.Build.DenseOfArray; 
            Target =
                reader.["data"].[sort].["targets"].Value 
                :?> double[,]  
                |> Matrix.Build.DenseOfArray
        }



    let ReadMatFile =
        let filename = @"C:\Users\Paul\Documents\NN_Assignments\Assignment3\data.mat"
        use file = File.OpenRead(filename)
        use reader = new MatReader(file)
        let extract = ExtractData reader

        {
            Training =
                "training" 
                |> extract;
            Validation = 
                "validation"
                |> extract;
            Test = 
                "test"
                |> extract
        }
