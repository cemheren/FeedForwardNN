using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using nnExample;

namespace nnTests
{
    [TestClass]
    public class GenericTests
    {
        [TestMethod]
        public void XNORApproximator()
        {
            double[][] trainingSet = new double[8][];
            trainingSet[0] = new double[] { 0, 0, 0 };
            trainingSet[1] = new double[] { 0, 0, 1 };
            trainingSet[2] = new double[] { 0, 1, 0 };
            trainingSet[3] = new double[] { 0, 1, 1 };
            trainingSet[4] = new double[] { 1, 0, 0 };
            trainingSet[5] = new double[] { 1, 0, 1 };
            trainingSet[6] = new double[] { 1, 1, 0 };
            trainingSet[7] = new double[] { 1, 1, 1 };

            double[] resultSet = new double[8] { 1, 0, 0, 1, 0, 1, 1, 0 }; //xnor

            var nn = new FeedForwardNetwork(3, 3, 1);

            for (int i = 0; i < 200000; i++)
            {
                var m = i % 8;
                nn.ForwardPass(trainingSet[m]);
                nn.BackPropagateForTarget(new double[] { resultSet[m] });
            }

            for (int i = 0; i < 8; i++)
            {
                nn.ForwardPass(trainingSet[i]);
                Assert.AreEqual(resultSet[i], Math.Round(nn.GetCurrentResult()[0]));
            }
        }

        [TestMethod]
        public void SineWaveApproximator()
        {
            var nn = new FeedForwardNetwork(1, 20, 1);
            var rand = new Random(13);

            for (int i = 0; i < 20000; i++)
            {
                var t = rand.NextDouble();
                nn.ForwardPass(new double[] { t });
                nn.BackPropagateForTarget(new double[] { Math.Sin(t) });
            }

            for (int i = 0; i < 10; i++)
            {
                var t = rand.NextDouble();
                nn.ForwardPass(new double[] { t });

                Assert.IsTrue(AreSimilar(Math.Sin(t), nn.GetCurrentResult()[0]));
            }
        }

        [TestMethod]
        public void MultiplicationApproximator()
        {
            var nn = new FeedForwardNetwork(2, 6, 1);
            var rand = new Random(13);

            for (int i = 0; i < 100000; i++)
            {
                var t1 = rand.NextDouble();
                var t2 = rand.NextDouble();

                nn.ForwardPass(new double[] { t1, t2 });
                nn.BackPropagateForTarget(new double[] { (t1 * t2) });
            }

            for (int i = 0; i < 10; i++)
            {
                var t1 = rand.NextDouble();
                var t2 = rand.NextDouble();

                nn.ForwardPass(new double[] { t1, t2 });

                Assert.IsTrue(AreSimilar(t1 * t2, nn.GetCurrentResult()[0]));
            }
        }

        [TestMethod]
        public void AdditionApproximator()
        {
            var nn = new FeedForwardNetwork(2, 6, 1);
            var rand = new Random(13);

            for (int i = 0; i < 100000; i++)
            {
                var t1 = rand.NextDouble() /2;
                var t2 = rand.NextDouble() /2;

                nn.ForwardPass(new double[] { t1, t2 });
                nn.BackPropagateForTarget(new double[] { (t1 + t2) });
            }

            for (int i = 0; i < 10; i++)
            {
                var t1 = rand.NextDouble() / 2;
                var t2 = rand.NextDouble() / 2;

                nn.ForwardPass(new double[] { t1, t2 });

                Assert.IsTrue(AreSimilar(t1 + t2, nn.GetCurrentResult()[0]));
            }
        }

        private bool AreSimilar(double f, double s)
        {
            return Math.Abs(f - s) < 0.1;
        }
    }
}
