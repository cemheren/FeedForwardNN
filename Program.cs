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

        public static double Sigmoid(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-1 * input));
        }

        public class XORNetwork
        {
            private Neuron[] inputNeurons;
            private Neuron[] hiddenNeurons;
            private Neuron[] outputNeurons;

            private Random rand = new Random(71);

            public double LearningRate
            {
                get
                {
                    return 0.3;
                    return (0.2 + (rand.NextDouble()));
                }
            }

            public XORNetwork(int input, int hidden)
            {
                input++;
                hidden++;

                inputNeurons = new Neuron[input];
                for (int i = 0; i < input; i++)
                {
                    inputNeurons[i] = new Neuron(0);
                }

                hiddenNeurons = new Neuron[hidden];
                for (int i = 0; i < hidden; i++)
                {
                    hiddenNeurons[i] = new Neuron(input);
                }

                outputNeurons = new Neuron[1];
                outputNeurons[0] = new Neuron(hidden);
            }

            public override string ToString()
            {
                var s = new StringBuilder();

                s.AppendLine();

                foreach (var neuron in hiddenNeurons)
                {
                    foreach (var weight in neuron.Weights)
                    {
                        s.Append(weight + " ");
                    }

                    s.Append("|| ");
                }

                s.AppendLine("------------------------------------------------------------------");

                foreach (var neuron in outputNeurons)
                {
                    foreach (var weight in neuron.Weights)
                    {
                        s.Append(weight + " ");
                    }

                    s.Append("|| ");
                }

                return s.ToString();
            }

            public double[] GetCurrentResult()
            {
                return outputNeurons.Select(n => n.currentValue).ToArray();
            }

            public void ForwardPass(double[] input)
            {
                // keep inputs in the network because we use them later in back propagation
                for (int i = 0; i < this.inputNeurons.Length - 1; i++)
                {
                    this.inputNeurons[i].currentValue = input[i];
                }
                this.inputNeurons.Last().currentValue = 1; // bias node

                for (int i = 0; i < hiddenNeurons.Length - 1; i++)
                {
                    hiddenNeurons[i].ActivateNeuron(this.inputNeurons.Select(n => n.currentValue).ToArray());
                }
                hiddenNeurons.Last().currentValue = 1; // bias node

                foreach (var neuron in outputNeurons)
                {
                    neuron.ActivateNeuron(hiddenNeurons.Select(n => n.currentValue).ToArray());
                }
            }

            public void BackPropagateForTarget(double[] target)
            {
                // Get the delta value for the output layer
                for (int i = 0; i < this.outputNeurons.Length; i++)
                {
                    this.outputNeurons[i].SetDeltaFromError(target[i] - this.outputNeurons[i].currentValue);
                }

                for (int i = 0; i < this.hiddenNeurons.Length; i++)
                {
                    double error = 0.0;
                    for (int j = 0; j < this.outputNeurons.Length; j++)
                    {
                        error += this.outputNeurons[j].Weights[i] * this.outputNeurons[j].CurrentMomentumErrorProduct;
                    }
                    this.hiddenNeurons[i].SetDeltaFromError(error);
                }
                // we propagated the errors back

                // Now update the weights between hidden & output layer
                UpdateWeights(this.outputNeurons, this.hiddenNeurons);

                // Now update the weights between input & hidden layer
                UpdateWeights(this.hiddenNeurons, this.inputNeurons);
            }

            private void UpdateWeights(Neuron[] outputLayer, Neuron[] inputLayer)
            {
                for (int i = 0; i < outputLayer.Length; i++)
                {
                    for (int j = 0; j < inputLayer.Length; j++)
                    {
                        outputLayer[i].Weights[j] += this.LearningRate * outputLayer[i].CurrentMomentumErrorProduct * inputLayer[j].currentValue;
                    }
                }
            }
        }

        public class Neuron
        {
            private Random rand = new Random(13);

            public double[] Weights;
            public double currentValue;

            public double CurrentMomentumErrorProduct;

            public Neuron()
            {
                Weights = new double[0];
            }
            public Neuron(int inputs)
            {
                Weights = new double[inputs];
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i] = (rand.NextDouble() * 4) - 2;
                }
            }

            public void ActivateNeuron(double[] input)
            {
                var sum = 0.0;
                for (int i = 0; i < this.Weights.Length; i++)
                {
                    sum += input[i] * this.Weights[i];
                }
                this.currentValue = Sigmoid(sum);
            }

            public void SetDeltaFromError(double error)
            {
                CurrentMomentumErrorProduct = currentValue * (1.0 - currentValue) * (error);
            }
        }

        public class Matrix
        {
            private double[,] m;

            public Matrix(int x, int y)
            {
                m = new double[x, y];
            }
        }
    }
}