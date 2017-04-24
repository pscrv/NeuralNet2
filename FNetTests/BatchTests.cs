using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using static FNet.BatchFunctions;
using static FNet.RecordTypes;
using static FNet.Exceptions;

namespace FNetTests
{
    [TestClass]
    public class BatchTests
    {
        [TestMethod]
        public void ExtractIndex0()
        {
            Matrix<double> input_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 }, { 0, 0, 0 }}
                );
            Matrix<double> target_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 } }
                );

            TrainingData td = new TrainingData(name: "test", input: input_matrix, target: target_matrix);
            TrainingData batch = MakeBatch(td, 0, 2);

            Assert.AreEqual(input_matrix.Column(0), batch.Input.Column(0));
            Assert.AreEqual(input_matrix.Column(1), batch.Input.Column(1));
            Assert.AreEqual(target_matrix.Column(0), batch.Target.Column(0));
            Assert.AreEqual(target_matrix.Column(1), batch.Target.Column(1));

        }

        [TestMethod]
        public void ExtractIndex1()
        {
            Matrix<double> input_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 }, { 0, 0, 0 } }
                );
            Matrix<double> target_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 } }
                );

            TrainingData td = new TrainingData(name: "test", input: input_matrix, target: target_matrix);
            TrainingData batch = MakeBatch(td, 1, 2);

            Assert.AreEqual(input_matrix.Column(2), batch.Input.Column(0));
            Assert.AreEqual(input_matrix.Column(0), batch.Input.Column(1));
            Assert.AreEqual(target_matrix.Column(2), batch.Target.Column(0));
            Assert.AreEqual(target_matrix.Column(0), batch.Target.Column(1));
        }

        [TestMethod]
        public void CannotExtractBadBatchsize()
        {
            Matrix<double> input_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 }, { 0, 0, 0 } }
                );
            Matrix<double> target_matrix = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 2, 3 } }
                );

            TrainingData td = new TrainingData(name: "test", input: input_matrix, target: target_matrix);

            try
            {
                TrainingData batch = MakeBatch(td, 1, 10);
                Assert.Fail("ExtractBatch failed to throw a BatchTooBigForDataSourceException.");
            }
            catch (BatchTooBigForDataSourceException) { }
        }
    }
}
