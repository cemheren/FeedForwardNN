using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    public class SineWaveApproximator
    {
        public static void M()
        {
            var nn = new FeedForwardNetwork(1, 20, 1);
            var rand = new Random(13);

            for (int i = 0; i < 50000; i++)
            {
                var t = rand.NextDouble();
                nn.ForwardPass(new double[] { t });
                nn.BackPropagateForTarget(new double[] { Math.Sin(t) });
            }

            for (int i = 0; i < 10; i++)
            {
                var t = rand.NextDouble();
                nn.ForwardPass(new double[] { t });

                Console.WriteLine(Math.Sin(t));
                Console.WriteLine(nn.GetCurrentResult()[0]);
                Console.WriteLine("==============================");
            }

            Console.WriteLine(nn.ToString());
        }
    }
}
