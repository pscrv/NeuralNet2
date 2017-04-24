namespace fsh_net

open System.IO
open Accord.IO
open MathNet.Numerics.LinearAlgebra

module DataExtraction =

    let ExtractData (reader: MatReader) (sort: string) =        
        let input = Matrix.Build.DenseOfArray( reader.["data"].[sort].["inputs"].Value :?> double[,] );
        let target = Matrix.Build.DenseOfArray( reader.["data"].[sort].["targets"].Value :?> double[,] );
        {Name = sort; Input = input; Target = target}



    let ReadMatFile =
        let filename = @"C:\Users\Paul\Documents\NN_Assignments\Assignment3\data.mat"
        let file = File.OpenRead(filename)
        use reader = new MatReader(file)
        {
            Training = ExtractData reader "training";
            Validation = ExtractData reader "validation";
            Test = ExtractData reader "test"
        }
