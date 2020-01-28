using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest
{
    class Activation
    {
        public static double ReLU(double val)
        {
            if(val > 0)
            {
                return val;
            }
            else
            {
                return 0;
            }
        }

        public static double Sigmoid(double val, bool derivative)
        {
            if (derivative)
            {
                double sigma = Sigmoid(val, false);
                return sigma * (1 - sigma);
            }
            else
            {
                return 1 / (1 + Math.Exp(-val));
            }
        }

        public static double TanH(double var)
        {
            return (2 / (1 + Math.Exp(-var))) - 1;
        }
    }
}
