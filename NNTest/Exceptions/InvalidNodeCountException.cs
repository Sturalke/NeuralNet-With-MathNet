using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest.Exceptions
{
    class InvalidNodeCountException : Exception
    {
        /// <summary>
        /// Exception is thrown when an invalid number of nodes is used to initialize a layer.
        /// </summary>
        /// <param name="Message">Message to be printed when thrown.</param>
        public InvalidNodeCountException(string Message) : base(Message)
        {
        }
    }
}
