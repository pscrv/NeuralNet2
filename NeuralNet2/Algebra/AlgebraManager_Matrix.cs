using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet.Algebra
{
    partial class AlgebraManager
    {
        public class Matrix
        {
            #region attributes
            protected Matrix<double> _denseMatrix;
            #endregion


            #region constructors
            protected Matrix() { }

            protected Matrix(Matrix<double> matrix)
            {
                 matrix.CopyTo(_denseMatrix);
            }


            public Matrix(int rows, int columns)
            {
                _denseMatrix = Matrix<double>.Build.Dense(rows, columns);
            }

            public Matrix(double[,] array)
            {
                _denseMatrix = Matrix<double>.Build.DenseOfArray(array);
            }
            #endregion



            #region properties
            public virtual int RowCount { get { return _denseMatrix.RowCount; } }
            public virtual int ColumnCount { get { return _denseMatrix.ColumnCount; } }
            #endregion


            #region public methods
            public virtual Matrix MultiplyBy(Matrix other)
            {
                Matrix<double> result = this._denseMatrix.Multiply(other._denseMatrix);
                return new Matrix(result);
            }
            #endregion


            #region protected methods
            public Vector<double> _getColumn(int index)
            {
                Vector<double> result = _denseMatrix.Column(index);
                return result;
            }
            #endregion


            #region static methods
            protected static Matrix MakeFromMatrix(Matrix<double> matrix)
            {
                return new Matrix(matrix);
            }
            #endregion
        }
    }
}
