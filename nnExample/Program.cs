using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainFile = File.ReadAllLines("../../../../tinyShakespeare/sonnets.txt").Take(5).ToArray().ToLowerInvariant();

            var dimensionVector = FindDifferentChars(trainFile);

            var flattened = trainFile.Flatten().Select(c => c.ConvertToOneHot(dimensionVector)).ToArray();

            var nn = new RecurrentNetwork(dimensionVector.Length, 10, dimensionVector.Length, 4);

            for (int i = 0; i < 1000; i++)
            {
                for (int k = 0; k < flattened.Length - 1; k++)
                {
                    nn.ForwardPass(flattened[k]);
                    nn.BackPropagateForTarget(flattened[k + 1]);
                }
            }

            var current = 'f'.ConvertToOneHot(dimensionVector);
            for (int i = 0; i < 100; i++)
            {
                Console.Write(dimensionVector[current.MaxIndex()]);

                nn.ForwardPass(current);
                current = nn.GetCurrentResult();
            }

            //Console.WriteLine(nn.ToString());

        }


        private static char[] FindDifferentChars(string[] s)
        {
            var hash = new HashSet<char>();

            int count = 0;
            for (int i = 0; i < s.Length; i++)
            {
                for (int k = 0; k < s[i].Length; k++)
                {
                    if (hash.Contains(s[i][k]) == false)
                    {
                        hash.Add(s[i][k]);
                        count++;
                    }
                }
            }

            return hash.ToArray();
        }
    }
}