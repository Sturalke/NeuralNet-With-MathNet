using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NNTest.Exceptions;
using MathNet.Numerics.LinearAlgebra;



namespace NNTest
{
    abstract class NodeLayer
    {
        //  ------------------------
        //  Begin Variables
        //  ------------------------

        /// <summary>
        /// Previous layer in the network.
        /// </summary>
        public NodeLayer Previous;
        
        /// <summary>
        /// Next layer in the network.
        /// </summary>
        public NodeLayer Next;


        /// <summary>
        /// Matrix in which the weights to the previous layer are stored.
        /// </summary>
        protected Matrix<float> _omega;

        /// <summary>
        /// Calculated delta values for each node during backprop.
        /// </summary>
        protected Matrix<float> _delta;

        /// <summary>
        /// Column vectors hold node values after the activation function.
        /// </summary>
        protected Matrix<float> _beta;

        /// <summary>
        /// Unmodified node values in column vectors.
        /// </summary>
        protected Matrix<float> _alpha;

        /// <summary>
        /// Number of nodes in this layer.
        /// </summary>
        public int Nodes;

        /// <summary>
        /// Number of passes to take at once.
        /// </summary>
        public int Passes;

        //  ------------------------
        //  End Variables
        //  ------------------------

        
        
        //  ------------------------
        //  Begin Constructors
        //  ------------------------

        /// <summary>
        /// Initialize variables.
        /// </summary>
        protected NodeLayer()
        {
            Previous = null;
            Next = null;
            _omega = null;
            _delta = null;
            _beta = null;
            _alpha = null;
            Nodes = 0;
            Passes = 1;
        }

        /// <summary>
        /// Gives a size to the matricies.
        /// </summary>>
        /// <param name="nodes">Number of nodes in this layer. (number of rows)</param>
        /// <param name="numPasses">Number of input passes. (number of columns)</param>
        protected NodeLayer(int nodes, int numPasses)
            : this()
        {
            if(nodes < 0)
            {
                throw new InvalidNodeCountException("Number of nodes cannot be less than zero.");
            }
            Nodes = nodes;

            if(numPasses < 1)
            {
                throw new InvalidPassCountException("Number of passes cannot be less than one.");
            }
            Passes = numPasses;

            _alpha = Matrix<float>.Build.Dense(nodes + 1, numPasses);
            _beta = Matrix<float>.Build.Dense(nodes + 1, numPasses);
        }

        /// <summary>
        /// Adds reference to the previous layer.
        /// </summary>
        /// <param name="nodes">Number of nodes in this layer. (number of rows)</param>
        /// <param name="numPasses">Number of input passes. (number of columns)</param>
        /// <param name="prev">Reference to the previous layer in the network.</param>
        protected NodeLayer(int nodes, int numPasses, NodeLayer prev)
            : this(nodes, numPasses)
        {
            Previous = prev;
        }

        /// <summary>
        /// Build the weights matrix using an initialization function.
        /// </summary>
        /// <param name="dist">Initialization function.</param>
        /// <remarks>
        /// <c>_omega</c> remains null if no previous layer is connected.
        /// Therefore we assume that this layer is considered an input layer.
        /// </remarks>
        public void BuildWeights(Func<int,int,Matrix<float>> dist)
        {
            if(Previous != null)    //  Check that this isn't an input layer
            {
                int nodes = Previous.Nodes + 1;
                _omega = dist(Nodes,nodes);
            }
        }

        //  ------------------------
        //  End Constructors
        //  ------------------------

        
        
        //  ------------------------
        //  Begin Methods
        //  ------------------------

        /// <summary>
        /// Activate the node values with the given activation function and store the result in another vector.
        /// </summary>
        /// <param name="activator">Activation function.</param>
        public void Activate(Func<float,float> activator)
        {
            _beta.Map(activator, _alpha);
        }

        /// <summary>
        /// Sets the number of nodes in this layer to a new value.
        /// Warning: resets all currently stored values and nulls the weights matrix.
        /// Use BuildWeights following this method.
        /// </summary>
        /// <param name="nodes">Number of nodes to be used in this layer.</param>
        public void SetNumNodes(int nodes)
        {
            if(nodes < 0)
            {
                throw new InvalidNodeCountException("Number of nodes cannot be less than zero.");
            }

            Nodes = nodes;
            _alpha = Matrix<float>.Build.Dense(Nodes + 1, Passes);
            _beta = Matrix<float>.Build.Dense(Nodes + 1, Passes);
            _omega = null;
        }

        /// <summary>
        /// Set the number of passes to new value.
        /// Warning: resets all currently stored values but leaves the weights alone.
        /// </summary>
        /// <param name="numPasses">New number of passes.</param>
        public void SetPasses(int numPasses)
        {
            if(numPasses < 1)
            {
                throw new InvalidPassCountException("Number of passes cannot be less than one.");
            }
            Passes = numPasses;
            _alpha = Matrix<float>.Build.Dense(Nodes + 1, Passes);
            _beta = Matrix<float>.Build.Dense(Nodes + 1, Passes);
        }

        /// <summary>
        /// Replaces weights matrix with the given one.
        /// Intended for adjusting weights via the calculated delta.
        /// </summary>
        /// <param name="weights">Given weight matrix.</param>
        public void SetWeights(Matrix<float> weights)
        {
            _omega = weights;
        }

        /// <summary>
        /// Retieve the weights matrix.
        /// </summary>
        /// <returns>Weight matrix.</returns>
        public Matrix<float> Weights()
        {
            return _omega;
        }

        //  ------------------------
        //  End Methods
        //  ------------------------
    }
}
