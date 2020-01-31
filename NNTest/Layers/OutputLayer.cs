using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    class OutputLayer : NodeLayer
    {

        public Matrix<float> Cost(Matrix<float> expected)
        {
            Matrix<float> cost = Matrix<float>.Build.Dense(this.Nodes,this.Passes,0f);
            Func<float, float, float> AbsCostFunc = (x, y) =>
            {
                return (float)Math.Pow(x - y, 2);
            };
            this._alpha.Map2(AbsCostFunc, expected, cost);
            return cost;
        }
    }
}
