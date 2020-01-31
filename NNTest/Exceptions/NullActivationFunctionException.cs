using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest.Exceptions
{
    class NullActivationFunctionException : Exception
    {
        /// <summary>
        /// Thrown when the Activation function for a layer is left unset.
        /// </summary>
        /// <param name="Message">Message to be shown when thrown.</param>
        public NullActivationFunctionException(string Message) : base(Message)
        {
        }
    }
}
