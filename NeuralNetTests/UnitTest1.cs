using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet.Algebra;

namespace NeuralNetTests
{
    [TestClass]
    public class Tryout
    {
        [TestMethod]
        public void Try()
        {
            NetMatrix a = AlgebraManager.MakeZeroWeightsMatrix(3, 2);
            NetMatrix b = AlgebraManager.MakeWeightsMatrixFromArray(new double[,] { { 1, 2, 3 }, { 2, 3, 4 } });
            NetMatrix c = AlgebraManager.MakeZeroBiasesVector(3);
            NetMatrix d = AlgebraManager.MakeVectorBatchFromArray(new double[,] { { 1, 2 }, { 2, 3 }, { 3, 4 } });
            NetMatrix e = AlgebraManager.MakeEmbeddingBatch(new int[,] { { 0, 1 }, { 1, 2 } }, 3);
        }
    }
}
