using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using FNet;

namespace FNetTests
{
    [TestClass]
    public class TrainingTests
    {

        [TestMethod]
        public void ModelGradients1()
        {
            Matrix<double> in_hid = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            Matrix<double> hid_class = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );
            
            Matrix<double> input = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            MutableTypes.Model networkModel = new MutableTypes.Model(in_hid, hid_class);
            MutableTypes.NetworkState networkState = NetRun.RunNetwork(networkModel, input);

            MutableTypes.Model gradients = Training._getModelGradients(networkModel, networkState, target);


            Matrix<double> hid_class_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0 } }
                );

            Matrix<double> in_hid_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0 } }
                );

            Assert.AreEqual(hid_class_gradient_check, gradients.HiddenToClassifier);
            Assert.AreEqual(in_hid_gradient_check, gradients.InputToHidden);
        }

        [TestMethod]
        public void ModelGradients2()
        {
            Matrix<double> in_hid = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            Matrix<double> hid_class = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 1 } }
                );

            Matrix<double> input = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 0 } }
                );

            MutableTypes.Model networkModel = new MutableTypes.Model(in_hid, hid_class);
            MutableTypes.NetworkState networkState = NetRun.RunNetwork(networkModel, input);

            MutableTypes.Model gradients = Training._getModelGradients(networkModel, networkState, target);


            Matrix<double> hid_class_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { - NetworkFunctions.Logistic(1) / 2 }, { NetworkFunctions.Logistic(1) / 2} }
                );

            Matrix<double> in_hid_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0 } }
                );

            Assert.AreEqual(hid_class_gradient_check, gradients.HiddenToClassifier);
            Assert.AreEqual(in_hid_gradient_check, gradients.InputToHidden);
        }

        [TestMethod]
        public void ModelGradients3()
        {
            Matrix<double> in_hid = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 2 } }
                );

            Matrix<double> hid_class = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1, 0 }, { 0, 1 } }
                );

            Matrix<double> input = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 } }
                );

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 0 } }
                );

            MutableTypes.Model networkModel = new MutableTypes.Model(in_hid, hid_class);
            MutableTypes.NetworkState networkState = NetRun.RunNetwork(networkModel, input);

            MutableTypes.Model gradients = Training._getModelGradients(networkModel, networkState, target);

            double l1 = NetworkFunctions.Logistic(1);
            double l2 = NetworkFunctions.Logistic(2);
            double el1 = Math.Exp(l1);
            double el2 = Math.Exp(l2);
            double N = el1 + el2;
            double el1N = el1 / N;
            double el2N = el2 / N;
            double ld1 = NetworkFunctions.LogisticDerivative_f_output(l1);
            double ld2 = NetworkFunctions.LogisticDerivative_f_output(l2);



            Matrix<double> hid_class_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { l1 * (el1N - 1), l2 * (el1N - 1) }, { l1 * el2N, l2 * el2N } }
                );

            Matrix<double> in_hid_gradient_check = Matrix<double>.Build.DenseOfArray(
                new double[,] { { ld1 * (el1N - 1 ) }, { ld2 * el2N} }
                );

            Assert.AreEqual(hid_class_gradient_check, gradients.HiddenToClassifier);
            Assert.AreEqual(in_hid_gradient_check, gradients.InputToHidden);
        }


        [TestMethod]
        public void ClassifierLoss1()
        {
            Matrix<double> classifier_input = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 0 } }
                );

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0.5 }, { 0.5 } }
                );

            Matrix<double> dummy = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0 } }
                );

            MutableTypes.Model model = new MutableTypes.Model(dummy, dummy);

            double loss = Training._getLoss(classifier_input, target, model, 0.0);

            double sm1 = Math.Exp(1);
            double sm2 = Math.Exp(0);
            double N = sm1 + sm2;
            double sm1N = sm1 / N;
            double sm2N = sm2 / N;

            double sm1N_log = - Math.Log(sm1N);
            double sm2N_log = - Math.Log(sm2N);
            double sm1N_log_target = sm1N_log * 0.5;
            double sm2N_log_target = sm2N_log * 0.5;
            double sum = sm1N_log_target + sm2N_log_target;

            Assert.AreEqual(sum, loss, 0.00000000001);            
        }


        [TestMethod]
        public void ClassifierLoss2()
        {
            Matrix<double> classifier_input = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 1 }, { 0.5 } }
                );

            Matrix<double> target = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 0.1 }, { 0.9 } }
                );

            Matrix<double> dummy = Matrix<double>.Build.DenseOfArray(
                new double[,] { { 2 } }
                );

            MutableTypes.Model model = new MutableTypes.Model(dummy, dummy);

            double loss = Training._getLoss(classifier_input, target, model, 0.1);

            double sm1 = Math.Exp(1);
            double sm2 = Math.Exp(0.5);
            double N = sm1 + sm2;
            double sm1N = sm1 / N;
            double sm2N = sm2 / N;

            double sm1N_log = -Math.Log(sm1N);
            double sm2N_log = -Math.Log(sm2N);
            double sm1N_log_target = sm1N_log * 0.1;
            double sm2N_log_target = sm2N_log * 0.9;
            double sum = sm1N_log_target + sm2N_log_target;

            double wd_loss = 0.1 * 8.0 / 2.0;

            Assert.AreEqual(sum + wd_loss, loss, 0.00000000001);
        }
    }
}
