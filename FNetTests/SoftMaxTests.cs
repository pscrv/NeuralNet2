using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using FNet;

namespace FNetTests
{
    [TestClass]
    public class SoftMaxTests
    {
        [TestMethod]
        public void GetSoftMaxOutput()
        {
            Matrix<double> matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 0 }, { 0, 2 } }
                );

            Matrix<double> softMax = SoftMax.GetSoftMaxOutput(matrix);

            double norm1 = Math.Exp(1) + Math.Exp(0);
            double norm2 = Math.Exp(0) + Math.Exp(2);
            Matrix<double> softmax_check = Matrix<double>.Build.DenseOfArray(
                new double[,] {
                    { Math.Exp(1) / norm1, Math.Exp(0) / norm2 },
                    { Math.Exp(0) / norm1, Math.Exp(2) / norm2 }
                }
                );

            for (int i = 0; i < softMax.RowCount; i++)
            {
                for (int j = 0; j < softMax.ColumnCount; j++)
                {
                    Assert.AreEqual(softmax_check[i, j], softMax[i, j], 0.00000001);
                }
            }
        }
        [TestMethod]
        public void GetSoftMaxLogOutput()
        {
            Matrix<double> matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 0 }, { 0, 2 } }
                );

            Matrix<double> softMax = SoftMax.GetSoftMaxLogOutput(matrix);

            double lognorm1 = Math.Log( Math.Exp(1) + Math.Exp(0) );
            double lognorm2 = Math.Log( Math.Exp(0) + Math.Exp(2) );
            Matrix<double> softmax_check = Matrix<double>.Build.DenseOfArray(
                new double[,] {
                    { 1 - lognorm1, 0 - lognorm2 },
                    { 0 - lognorm1, 2 - lognorm2 }
                }
                );

            for (int i = 0; i < softMax.RowCount; i++)
            {
                for (int j = 0; j < softMax.ColumnCount; j++)
                {
                    Assert.AreEqual(softmax_check[i, j], softMax[i, j], 0.00000001);
                }

            }
        }

    }
}
