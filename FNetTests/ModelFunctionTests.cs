using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using static FNet.ModelFunctions;
using static FNet.RecordTypes;
using static FNet.MutableTypes;
using MathNet.Numerics.LinearAlgebra;

namespace FNetTests
{
    [TestClass]
    public class ModelFunctionTests
    {
        [TestMethod]
        public void GetZero()
        {
            Parameters parameters = new Parameters
            (
                numberOfInputs: 2,
                numberOfHiddenUnits: 3,
                numberOfClasses: 2,
                numberOfIterations: 0,
                batchSize: 0,
                learningRate: 0.0,
                momentumCoefficient: 0.0,
                weightDecayCoefficient: 0.0
            );

            Model model = GetZeroModel(parameters);

            Matrix<double> _in_to_hid_test = Matrix<double>.Build.Dense(3, 2);
            Matrix<double> _hid_to_class_test = Matrix<double>.Build.Dense(2, 3);

            Assert.AreEqual(_in_to_hid_test, model.InputToHidden);
            Assert.AreEqual(_hid_to_class_test, model.HiddenToClassifier);
        }

        [TestMethod]
        public void Initialise()
        {
            Parameters parameters = new Parameters
            (
                numberOfInputs: 2,
                numberOfHiddenUnits: 3,
                numberOfClasses: 2,
                numberOfIterations: 0,
                batchSize: 0,
                learningRate: 0.0,
                momentumCoefficient: 0.0,
                weightDecayCoefficient: 0.0
            );

            Model model = InitialiseModel(parameters);

            Matrix<double> _in_to_hid_test = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { Math.Cos(0 + (0 * 3)), Math.Cos(0 + (1 * 3)) },
                    { Math.Cos(1 + (0 * 3)), Math.Cos(1 + (1 * 3)) },
                    { Math.Cos(2 + (0 * 3)), Math.Cos(2 + (1 * 3)) }
                }
                );

            Matrix<double> _hid_to_class_test = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { Math.Cos(6 + 0 + (0 * 2)), Math.Cos(6 + 0 + (1 * 2)), Math.Cos(6 + 0 + (2 * 2)) },
                    { Math.Cos(6 + 1 + (0 * 2)), Math.Cos(6 + 1 + (1 * 2)), Math.Cos(6 + 1 + (2 * 2)) }
                }
                );

            Assert.AreEqual(_in_to_hid_test, model.InputToHidden);
            Assert.AreEqual(_hid_to_class_test, model.HiddenToClassifier);
        }
    }
}
