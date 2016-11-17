using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
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

        public void SetMomentumErrorProduct(double error)
        {
            CurrentMomentumErrorProduct = currentValue * (1.0 - currentValue) * (error);
        }

        public static double Sigmoid(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-1 * input));
        }
    }
}
