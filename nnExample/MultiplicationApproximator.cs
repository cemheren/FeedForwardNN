using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    public class MultiplicationApproximator
    {
        public static void M()
        {
            var nn = new FeedForwardNetwork(2, 6, 1);
            var rand = new Random(13);

            for (int i = 0; i < 500000; i++)
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

                Console.WriteLine((t1 * t2));
                Console.WriteLine(nn.GetCurrentResult()[0]);
                Console.WriteLine("==============================");
            }

            Console.WriteLine(nn.ToString());
        }
    }
}
