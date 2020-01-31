using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    class Network
    {
        private List<NodeLayer> layers;
        private OutputLayer Output;

        private void ComputeDeltas(Matrix<float> target)
        {
            Matrix<float> a = Output._beta.Clone();

            //  Perform the derivative of the activation function on the values of the nodes
            Matrix<float> b = Output.ActivationFunctionDerivative(Output._alpha);

            // Adds the negative of each column vector in the target matrix to the respective column vector in ______
            //a.EnumerateColumnsIndexed().Select(t => t.Item2.Add(target.Column(t.Item1).Negate()));
            a = a - target;

            Matrix<float> OutLayerDeltas = a.PointwiseMultiply(b);

            Output._delta = OutLayerDeltas;



            NodeLayer current = Output.Previous;
            bool inputLayer = false;
            
            while(!inputLayer)
            {
                Matrix<float> transWeight = current.Next.Weights().TransposeThisAndMultiply(current.Next._delta);

                // MISSING CODE HERE

                if(current.Previous == null)
                {
                    inputLayer = true;
                }
            }
        }
    }
}
