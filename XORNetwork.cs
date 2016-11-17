using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
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
}
