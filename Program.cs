using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    class Program
    {
        static void Main(string[] args)
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

            var nn = new XORNetwork(3, 3);

            for (int i = 0; i < 20000; i++)
            {
                var m = i % 8;
                nn.ForwardPass(trainingSet[m]);
                nn.BackPropagateForTarget(new double[] { resultSet[m] });
            }

            for (int i = 0; i < 8; i++)
            {
                nn.ForwardPass(trainingSet[i]);
                Console.WriteLine(nn.GetCurrentResult()[0]);
            }

            Console.WriteLine(nn.ToString());
        }
    }
}