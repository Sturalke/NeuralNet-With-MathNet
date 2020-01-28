using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace NNTest
{
    class Initialization
    {

        public static Matrix<float> Xavier(int rows, int columns)
        {
            return Matrix<float>.Build.Dense(rows,columns);
        }
    }
}
