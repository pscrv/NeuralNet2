using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public static partial class AlgebraManager
    {
        #region attributes
        private static Dictionary<MatrixIdentifier, Matrix> __matrixStore = new Dictionary<MatrixIdentifier, Matrix>();
        #endregion


        #region Factory methods
        public static WeightsMatrix MakeZeroWeightsMatrix(int rows, int columns)
        {
            MatrixIdentifier id = __makeMatrix(rows, columns);
            return new WeightsMatrix(id);
        }

        public static WeightsMatrix MakeWeightsMatrixFromArray(double[,] array)
        {
            MatrixIdentifier id = __makeMatrix(array);
            return new WeightsMatrix(id);
        }

        public static BiasesVector MakeZeroBiasesVector(int dimension)
        {
            MatrixIdentifier id = __makeMatrix(1, dimension);
            return new BiasesVector(id);
        }

        public static VectorBatch MakeVectorBatchFromArray(double[,] array) 
        {
            MatrixIdentifier id = __makeMatrix(array);
            return new VectorBatch(id);
        }




        public static EmbeddingBatch MakeEmbeddingBatch(int[,] array, int partDimension) 
        {

            Matrix matrix = new EmbeddingMatrix(array, partDimension);
            MatrixIdentifier id = __storeAndGetID(matrix);
            return new EmbeddingBatch(id);
        }



        #endregion


        #region private utility methods
        private static MatrixIdentifier __makeMatrix(int rows, int columns)
        {
            Matrix matrix = new Matrix(rows, columns);
            MatrixIdentifier id = __storeAndGetID(matrix);
            return id;
        }

        private static MatrixIdentifier __makeMatrix(double[,] array)
        {
            Matrix matrix = new Matrix(array);
            MatrixIdentifier id = __storeAndGetID(matrix);
            return id;
        }

        private static MatrixIdentifier __storeAndGetID(Matrix matrix)
        {
            MatrixIdentifier id = new MatrixIdentifier();
            __matrixStore[id] = matrix;
            return id;
        }
        #endregion



        #region WeightedCombining
        public static VectorBatch __ApplyWeightsToVectorBatch(WeightsMatrix weightsMatrix, VectorBatch inputBatch)
        {
            Matrix matrix = __matrixStore[weightsMatrix.MatrixID];
            Matrix batch = __matrixStore[inputBatch.MatrixID];
            Matrix resultMatrix = batch.MultiplyBy(matrix);

            MatrixIdentifier id = __storeAndGetID( resultMatrix);
            return new VectorBatch(id);
        }
        #endregion


        #region general public matrix methods
        public static int GetRowCount(MatrixIdentifier id)
        {
            return __matrixStore[id].RowCount;
        }

        public static int GetColumnCount(MatrixIdentifier id)
        {
            return __matrixStore[id].ColumnCount;
        }
        #endregion


    }
}
