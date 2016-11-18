using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    public class MNISTClassifier
    {
        public static void M()
        { 
            var trainFile = File.ReadAllLines("../../mnist_train.csv");
            var testFile = File.ReadAllLines("../../mnist_test.csv");

            var parsedFile = ParseFile(trainFile.Take(60000).ToArray());
            var trainData = parsedFile.Item1;
            var trainLabel = parsedFile.Item2;

            var parsed = ParseFile(testFile.Take(10000).ToArray());
            var testData = parsedFile.Item1;
            var testLabel = parsedFile.Item2;

            Console.WriteLine("Read all the data");

            var nn = new FeedForwardNetwork(784, 15, 10);
            var rand = new Random(13);

            for (int e = 0; e < 100; e++)
            {
                for (int i = 0; i < 55000; i++)
                {
                    nn.ForwardPass(trainData[i]);
                    //var a = nn.GetCurrentResult();
                    //Console.WriteLine(a.ToNiceString());
                    //Console.WriteLine("======" + MaxIndex(trainLabel[i]).ToString() + "======");
                    nn.BackPropagateForTarget(trainLabel[i]);
                }

                var currentMatchCount = 0;

                for (int i = 55000; i < 60000; i++)
                {
                    nn.ForwardPass(trainData[i]);
                    var currentPrediction = nn.GetCurrentResult();
                    var p = MaxIndex(currentPrediction);
                    var r = MaxIndex(trainLabel[i]);

                    if (p == r)
                    {
                        currentMatchCount++;       
                    }
                }

                Console.WriteLine("current acc: " + currentMatchCount / 5000.0);
                Console.WriteLine("Epoch " + e + " completed");
            }

            var cMatchCount = 0;
            for (int i = 0; i < 10000; i++)
            {
                nn.ForwardPass(testData[i]);

                var currentPrediction = nn.GetCurrentResult();
                var p = MaxIndex(currentPrediction);
                var r = MaxIndex(testLabel[i]);

                if (p == r)
                {
                    cMatchCount++;
                }

                Console.WriteLine(r);
                Console.WriteLine(p);

                Console.WriteLine("=======================");
                //nn.BackPropagateForTarget(trainLabel[i]);
            }

            Console.WriteLine("mc "+ cMatchCount);
        }

        public static Tuple<double[][], double[][]> ParseFile(string[] text)
        {
            var data = new double[text.Length][];
            var label = new double[text.Length][]; // represent this in 1-hot encoding

            Parallel.For(0, text.Length, (i) =>
            {
                var csv = text[i].Split(',');
                data[i] = csv.Skip(1).Select(n => double.Parse(n) / 255.0).ToArray(); // normalize
                label[i] = ConvertToOneHot(int.Parse(csv[0]));
            });

            return new Tuple<double[][], double[][]>(data, label);
        }

        private static double[] ConvertToOneHot(int d) 
        {
            var result = new double[10];

            for (int i = 0; i < 10; i++)
            {
                if (i == d)
                {
                    result[i] = 1;
                    return result;
                }
            }

            return result;
        }

        private static int MaxIndex(double[] arr)
        {
            double maxValue = arr.Max();
            int maxIndex = arr.ToList().IndexOf(maxValue);
            return maxIndex;
        }
    }

    public static class Print
    {
        public static string ToNiceString(this double[] x)
        {
            return string.Join("\n", x);
        }
    }
}
