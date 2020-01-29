using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest.Exceptions
{
    class InvalidInputException : Exception
    {
        /// <summary>
        /// Thrown when the input is determined to be invalid.
        /// Typically when the input matrix is the wrong size.
        /// </summary>
        /// <param name="Message">Message to be shown when thrown.</param>
        public InvalidInputException(string Message) : base(Message)
        {
        }
    }
}
