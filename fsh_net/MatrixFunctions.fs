namespace fsh_net

open MathNet.Numerics.LinearAlgebra

[<AutoOpen>]
module MatrixFunctions = 

    let ColumnMaxima (matrix: Matrix<double>) =
        matrix.ColumnNorms(System.Double.PositiveInfinity)

    let SumElements (matrix: Matrix<double>) =
        matrix.RowSums().Sum()

    let SubtractVectorFromEachRow (matrix: Matrix<double>) (vector: Vector<double>) =
        let mutable result = Matrix.Build.DenseOfMatrix(matrix)
        for (index, row) in matrix.EnumerateRowsIndexed() do
            result.SetRow(index, matrix.Row(index).Subtract(vector))
        result

