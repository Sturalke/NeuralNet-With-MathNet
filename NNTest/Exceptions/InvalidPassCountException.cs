using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest.Exceptions
{
    class InvalidPassCountException : Exception
    {
        /// <summary>
        /// Throws when the number of passes is invalid.
        /// </summary>
        /// <param name="Message">Message to be displayed when thrown.</param>
        public InvalidPassCountException(string Message) : base(Message)
        {
        }
    }
}
