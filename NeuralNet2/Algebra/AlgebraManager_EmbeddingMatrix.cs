using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public static partial class AlgebraManager
    {

        class EmbeddingMatrix : Matrix
        {
            #region attributes
            int[,] _matrixStore;
            int _dimension;
            int _partCount;
            int _vectorCount;
            #endregion


            #region constructors
            public EmbeddingMatrix(int[,] array, int partDimension)
            {
                for (int i = 0; i < array.GetLength(0); i++)
                    for (int j = 0; j < array.GetLength(1); j++)
                    {
                        if (array[i, j] >= partDimension)
                            throw new IndexOutOfRangeException();
                    }
                
                _matrixStore = null;
                _matrixStore = array.Clone() as int[,];
                _vectorCount = array.GetLength(0);
                _partCount = array.GetLength(1);
                _dimension = _partCount * partDimension;
            }

            private EmbeddingMatrix(Matrix<double> matrix)
                : base (matrix)
            { }

            #endregion


            #region properties
            public override int ColumnCount { get { return _dimension; } }
            public override int RowCount { get { return _vectorCount; } }
            #endregion



            #region public methods
            public override Matrix MultiplyBy(Matrix other)
            {
                Matrix<double> result = Matrix<double>.Build.Dense(_vectorCount, _dimension);
                List<double> vector = new List<double>();
                Vector<double> workingVector;

                for (int rowIndex = 0; rowIndex < _vectorCount; rowIndex++)
                {
                    for (int colIndex = 0; colIndex < _partCount; colIndex++)
                    {
                        workingVector = other._getColumn(_matrixStore[rowIndex, colIndex]);
                        vector.AddRange(workingVector);
                    }
                                        
                    result.SetRow(rowIndex, Vector<double>.Build.DenseOfEnumerable(vector));
                }

                Matrix resultMatrix = Matrix.MakeFromMatrix(result);
                return resultMatrix;
            }
            #endregion



            public Matrix AsMatrix()  // Does this actually work?
            {
                _denseMatrix = Matrix<double>.Build.Dense(RowCount, ColumnCount);
                for (int rowIndex = 0; rowIndex < RowCount; rowIndex++)
                {
                    for (int colIndex = 0; colIndex < ColumnCount; colIndex++)
                    {
                        _denseMatrix[rowIndex, _matrixStore[rowIndex, colIndex]] = 1.0;
                    }
                }

                return this as Matrix;
            }
        }
    }
}
