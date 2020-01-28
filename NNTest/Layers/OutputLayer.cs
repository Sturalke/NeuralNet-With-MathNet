using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    class OutputLayer : Layer
    {

        public override void PropagateFromPrevious()
        {
            throw new NotImplementedException();
        }

        public Vector<float> Cost(Vector<float> expected)
        {
            Vector<float> costVect = Vector<float>.Build.Dense(this.Nodes,0f);
            Func<float, float, float> AbsCostFunc = (x, y) =>
            {
                return (float)Math.Pow(x - y, 2);
            };
            this._alpha.Map2(AbsCostFunc, expected, costVect);
            return costVect;
        }
    }
}
