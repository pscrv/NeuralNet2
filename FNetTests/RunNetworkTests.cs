using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using static FNet.NetRun;
using static FNet.MutableTypes;
using static FNet.NetworkFunctions;

namespace FNetTests
{
    [TestClass]
    public class RunNetworkTests
    {
        [TestMethod]
        public void CanRun()
        {
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1, 2 }, { 1, 2, 3 } });
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 1 }, { 1, 2 }, { 2, 3 } });

            Matrix<double> batch_input = Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });

            Model model = new Model(m1, m2);
            NetworkState state = RunNetwork(model, batch_input);

            Matrix<double> hiddenInputCheck = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0, 1, 2 }, { 1, 2, 3 } }
                );
            Matrix<double> hiddenOutputCheck = hiddenInputCheck.Map(Logistic);
            Matrix<double> classifierInputCheck = m2.Multiply(hiddenOutputCheck);
            int[] choiceCheck = new[] { 2, 2, 2 };

            Assert.AreEqual(batch_input, state.Input);
            Assert.AreEqual(hiddenInputCheck, state.HiddenInput);
            Assert.AreEqual(hiddenOutputCheck, state.HiddenOutput);
            Assert.AreEqual(classifierInputCheck, state.ClassifierInput);

            for (int i = 0; i < choiceCheck.Length; i++)
            {
                Assert.AreEqual(choiceCheck[i], state.ClassifierChoices[i]);
            }

        }
    }
}
