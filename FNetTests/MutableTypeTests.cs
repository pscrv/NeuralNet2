using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using static FNet.MutableTypes;
using MathNet.Numerics.LinearAlgebra;
using static FNet.Exceptions;

namespace FNetTests
{
    [TestClass]
    public class LossesTests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                Losses losses = new Losses();
            }
            catch (Exception e)
            {
                Assert.Fail("Losses constructor threw an exception: " + e.Message);
            }      
        }

        [TestMethod]
        public void CanAdd()
        {
            Losses losses = new Losses();
            losses.AddLossPair(1.0, 2.0);

            Assert.AreEqual(1, losses.TrainingLosses.Length);
            Assert.AreEqual(1, losses.ValidationLosses.Length);
            Assert.AreEqual(1.0, losses.LatestTrainingLoss);
            Assert.AreEqual(2.0, losses.LatestValidationLoss);
        }
    }

    [TestClass]
    public class ModelTests
    {
        [TestMethod]
        public void CanMake()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });

            try
            {
                Model model = new Model(m1, m2);
            }
            catch (Exception e)
            {
                Assert.Fail("Losses constructor threw an exception: " + e.Message);
            }
        }

        [TestMethod]
        public void CannotMakeMismatched()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });

            try
            {
                Model model = new Model(m1, m2);
                Assert.Fail("Losses constructor failed to throw an InconsistentModelException: ");
            }
            catch (InconsistentModelException)
            {
            }
        }

        [TestMethod]
        public void CanDispose()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);

            try
            {
                (model as IDisposable).Dispose();
            }
            catch (Exception e)
            {
                Assert.Fail("Failed to dispose, with exception: " + e.Message);
            }
        }

        [TestMethod]
        public void CanScale()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);

            try
            {
                model.Scale(2.0);
            }
            catch (Exception e)
            {
                Assert.Fail("Failed to scale, with exception: " + e.Message);
            }

            Assert.AreEqual(0.0, model.InputToHidden[0, 0]);
            Assert.AreEqual(2.0, model.InputToHidden[1, 0]);
            Assert.AreEqual(6.0, model.InputToHidden[1, 2]);
            Assert.AreEqual(0.0, model.HiddenToClassifier[0, 0]);
            Assert.AreEqual(2.0, model.HiddenToClassifier[1, 0]);
            Assert.AreEqual(6.0, model.HiddenToClassifier[2, 1]);
        }

        [TestMethod]
        public void CanSubtract()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);

            try
            {
                model.Subtract(model);
            }
            catch (Exception e)
            {
                Assert.Fail("Failed to subtract, with exception: " + e.Message);
            }

            Assert.AreEqual(0.0, model.InputToHidden[0, 0]);
            Assert.AreEqual(0.0, model.InputToHidden[1, 0]);
            Assert.AreEqual(0.0, model.InputToHidden[1, 2]);
            Assert.AreEqual(0.0, model.HiddenToClassifier[0, 0]);
            Assert.AreEqual(0.0, model.HiddenToClassifier[1, 0]);
            Assert.AreEqual(0.0, model.HiddenToClassifier[2, 1]);
        }

        [TestMethod]
        public void CanAdd()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);

            try
            {
                model.Add(model);
            }
            catch (Exception e)
            {
                Assert.Fail("Failed to subtract, with exception: " + e.Message);
            }

            Assert.AreEqual(0.0, model.InputToHidden[0, 0]);
            Assert.AreEqual(2.0, model.InputToHidden[1, 0]);
            Assert.AreEqual(6.0, model.InputToHidden[1, 2]);
            Assert.AreEqual(0.0, model.HiddenToClassifier[0, 0]);
            Assert.AreEqual(2.0, model.HiddenToClassifier[1, 0]);
            Assert.AreEqual(6.0, model.HiddenToClassifier[2, 1]);
        }
        
        [TestMethod]
        public void CanSumOfSquares()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);

            Assert.AreEqual(38.0, model.SumOfSquares);
        }

        [TestMethod]
        public void CanCopy()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });
            Model model = new Model(m1, m2);
            Model modelCopy = null;

            try
            {
               modelCopy = model.Copy;
            }
            catch (Exception e)
            {
                Assert.Fail("Failed to copy, with exception: " + e.Message);
            }

            Assert.AreEqual(model.InputToHidden, modelCopy.InputToHidden);
            Assert.AreEqual(model.HiddenToClassifier, modelCopy.HiddenToClassifier);
        }
    }   

}
