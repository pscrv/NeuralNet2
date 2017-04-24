namespace FNet

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module MatrixFunctions = 

    let ColumnMaxima (matrix: Matrix<double>) =
        matrix.ColumnNorms(System.Double.PositiveInfinity)
        
    let ColumnSums (matrix: Matrix<double>) =
        matrix.ColumnSums()

    let NormaliseColumns (matrix : Matrix<double>) =
        matrix.NormalizeColumns 1.0

    let SumVector (vector : Vector<double>) =
        vector.Sum()

    let VectorElementMean (vector : Vector<double>) =
        (vector
        |> SumVector) / (double vector.Count)

    let SubtractVectorFromEachRow (matrix: Matrix<double>) (vector: Vector<double>) =
        let mutable result = Matrix.Build.DenseOfMatrix(matrix)
        for (index, row) in matrix.EnumerateRowsIndexed() do
            result.SetRow(index, matrix.Row(index).Subtract(vector))
        result

