using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNTest.Exceptions;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    class InputLayer : NodeLayer
    {
        //  ------------------------
        //  Begin Constructors
        //  ------------------------

        //  Default constructor hidden

        /// <summary>
        /// Sets the number of nodes in this layer. Passes defaults to one.
        /// </summary>
        /// <param name="nodes">Number of nodes in the input.</param>
        public InputLayer(int nodes)
            : base(nodes, 1)
        {
        }

        /// <summary>
        /// Sets the number of nodes and the number of passes.
        /// </summary>
        /// <param name="nodes">Number of nodes in the input.</param>
        /// <param name="passes">Number of passes to be performed.</param>
        public InputLayer(int nodes, int passes)
            : base(nodes, passes)
        {
        }

        //  Reference constructor hidden

        //  ------------------------
        //  End Constructors
        //  ------------------------



        //  ------------------------
        //  Begin Methods
        //  ------------------------

        /// <summary>
        /// Sets the inputs to the node values.
        /// Appends a row of ones to the end of the matrix to act as the "bias".
        /// </summary>
        /// <param name="inputs">Matrix of inputs.</param>
        public void SetInputs(Matrix<float> inputs)
        {
            if(inputs.RowCount != this.Nodes)
            {
                throw new InvalidInputException("Input length isn't consistant with the number of nodes in this layer.");
            }

            if(inputs.ColumnCount != this.Passes)
            {
                throw new InvalidInputException("Input columns isn't consistant with the number of passes.");
            }

            Matrix<float> temp = Matrix<float>.Build.Dense(inputs.RowCount, inputs.ColumnCount);
            inputs.CopyTo(temp);
            temp.InsertRow(inputs.RowCount, Vector<float>.Build.Dense(inputs.ColumnCount, 1f));

            _alpha = temp;
        }

        //  ------------------------
        //  End Methods
        //  ------------------------
    }
}
