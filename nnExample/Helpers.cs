using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnExample
{
    public static class Helpers
    {
        public static string[] ToLowerInvariant(this string[] text)
        {
            for (int i = 0; i < text.Length; i++)
            {
                text[i] = text[i].ToLowerInvariant();
            }
            return text;
        }

        public static double[] ConvertToOneHot(this int integer, int dimension)
        {
            var result = new double[dimension];

            for (int i = 0; i < dimension; i++)
            {
                if (i == integer)
                {
                    result[i] = 1;
                    return result;
                }
            }

            return result;
        }

        public static double[] ConvertToOneHot(this char c, char[] dimensionVector)
        {
            var result = new double[dimensionVector.Length];

            for (int i = 0; i < dimensionVector.Length; i++)
            {
                if (c == dimensionVector[i])
                {
                    result[i] = 1;
                    return result;
                }
            }

            return result;
        }

        public static int MaxIndex(this double[] arr)
        {
            double maxValue = arr.Max();
            int maxIndex = arr.ToList().IndexOf(maxValue);
            return maxIndex;
        }

        public static string ToNiceString(this double[] x)
        {
            return string.Join("\n", x);
        }

        public static double Sigmoid(this double input)
        {
            return 1.0 / (1.0 + Math.Exp(-1 * input));
        }

        public static T[] Flatten<T>(this T[][] arr)
        {
            var result = new List<T>();

            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    result.Add(arr[i][k]);
                }
            }

            return result.ToArray();
        }
        public static char[] Flatten(this string[] arr)
        {
            var result = new List<char>();

            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    result.Add(arr[i][k]);
                }
            }

            return result.ToArray();
        }
    }
}
