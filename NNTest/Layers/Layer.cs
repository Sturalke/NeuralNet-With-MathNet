using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;



namespace NNTest
{
    abstract class Layer
    {
        public Layer Previous;
        public Layer Next;

        protected Matrix<float> _omega;
        protected Vector<float> _beta;
        protected Vector<float> _alpha;

        public int In;
        public int Out;
        public int Nodes;

        protected Layer()
        {
            Previous = null;
            Next = null;
            _omega = null;
            In = 0;
            Out = 0;
        }
        protected Layer(int nodes)
            : this()
        {
            this.Nodes = nodes;
            _alpha = Vector<float>.Build.Dense(nodes + 1, 0f);
            _beta = Vector<float>.Build.Dense(nodes + 1, 0f);
        }
        protected Layer(int nodes, Layer prev)
            : this(nodes)
        {
            Previous = prev;
            In = (Nodes + 1) * (prev.Nodes + 1);
            prev.Out = In;
        }

        public void BuildWeights(Func<int,int,float> dist)
        {
            if(Previous != null)
            {
                int nodes = Previous.Nodes + 1;
                _omega = Matrix<float>.Build.Dense(nodes, nodes, dist);
            }
        }
        
        public void Activate(Func<float,float> activator)
        {
            _beta.Map(activator, _alpha);
        }

        public abstract void PropagateFromPrevious();
        
    }
}
