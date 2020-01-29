using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace NNTest
{
    /// <summary>
    /// Collection of weight initialization methods.
    /// </summary>
    class Initialization
    {
        /// <summary>
        /// Utilizes initialization method proposed and tested by Xavier Glorot and Yoshua Bengio
        /// in their paper titled
        /// "Understanding the difficulty of training deep feedforward neural networks."
        /// </summary>
        /// <param name="rows">Number of nodes in the current layer.</param>
        /// <param name="columns">Number of nodes in the previous layer plus one.</param>
        /// <returns>Matrix of weight values to be applied to the network</returns>
        /// <remarks>
        /// To be finished.
        /// </remarks>
        public static Matrix<float> Xavier(int rows, int columns, int fan_in, int fan_out)
        {
            return Matrix<float>.Build.Random(rows,columns, new Normal(0, Math.Sqrt( 2/(fan_in + fan_out) )));
        }

        public static Matrix<float> Kaiming(int rows, int columns, int fan_in)
        {
            return Matrix<float>.Build.Random(rows, columns, new Normal(0, Math.Sqrt(1/fan_in) ));
        }
    }
}
