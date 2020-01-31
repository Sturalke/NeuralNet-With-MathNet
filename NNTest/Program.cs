using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Matrix<float> a = Matrix<float>.Build.Dense(3, 1, (int r, int c) =>
              {
                  if(r == 0)
                  {
                      return 1f;
                  }
                  else if(r == 1)
                  {
                      return 0f;
                  }
                  else
                  {
                      return -18f;
                  }
              });

            //float max = float.MinValue;
            //foreach (float f in a.Enumerate())
            //{
            //    max = Math.Max(f, max);
            //}

            //a.MapInplace((float n) =>
            //{
            //    return n - max;
            //});

            Matrix<float> soft = Activation.SoftMax(a);

            Matrix<float> softDer = Activation.SoftMaxDerivative(a,a.MapIndexed((int row, int column, float n) =>
            {
                if (row == 0)
                {
                    return 0f;
                }
                else if(row == 1)
                {
                    return 0f;
                }
                else
                {
                    return 1f;
                }
            }),
            (float c, float b) =>
            {
                return c - b;
            });

            Matrix<float> jac = Activation.SoftMaxJacobian(a.Column(0), soft.Column(0));

            Console.WriteLine(a.ToString());
            Console.WriteLine(soft.ToString());
            Console.WriteLine(softDer.ToString());
            Console.ReadKey();
        }
    }
}
