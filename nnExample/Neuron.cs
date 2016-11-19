using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    public class Neuron
    {
        protected Random rand = new Random(13);

        public double[] InputWeights;
        public double currentValue;

        public double CurrentMomentumErrorProduct;

        public Neuron()
        {
            InputWeights = new double[0];
        }
        public Neuron(int inputs)
        {
            InputWeights = new double[inputs];
            for (int i = 0; i < InputWeights.Length; i++)
            {
                InputWeights[i] = (rand.NextDouble() * 4) - 2;
            }
        }

        public void ActivateNeuron(double[] input)
        {
            var sum = 0.0;
            for (int i = 0; i < this.InputWeights.Length; i++)
            {
                sum += input[i] * this.InputWeights[i];
            }
            this.currentValue = Helpers.Sigmoid(sum);
        }

        public void SetMomentumErrorProduct(double error)
        {
            CurrentMomentumErrorProduct = currentValue * (1.0 - currentValue) * (error);
        }
    }

    public class RecurrentNeuron : Neuron
    {
        public double[][] SelfWeightsThroughTime;
        private int lookback;

        public RecurrentNeuron(int numberOfInputs, int ownLayerSize, int lookbackInTime = 1) : base(numberOfInputs)
        {
            this.lookback = lookbackInTime;

            SelfWeightsThroughTime = new double[ownLayerSize][];
            for (int i = 0; i < ownLayerSize; i++)
            {
                SelfWeightsThroughTime[i] = new double[lookbackInTime];
                for (int k = 0; k < lookbackInTime; k++)
                {
                    SelfWeightsThroughTime[i][k] = (base.rand.NextDouble() * 4) - 2;
                }
            }
        }

        public void ActivateRecurrentNeuron(double[] input, double[] selfLayerValues)
        {
            var sum = 0.0;
            for (int i = 0; i < this.InputWeights.Length; i++)
            {
                sum += input[i] * this.InputWeights[i];
            }
            
            for (int i = 0; i < this.SelfWeightsThroughTime.Length; i++)
            {
                for (int t = 0; t < this.lookback; t++)
			    {
                    sum += selfLayerValues[i] * this.SelfWeightsThroughTime[i][t];
                } 
			}

            this.currentValue = Helpers.Sigmoid(sum);
        }
    }
}
