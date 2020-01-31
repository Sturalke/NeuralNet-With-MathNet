using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NNTest
{
    /// <summary>
    /// Collection of activation functions to be performed on a node's value before a propogation step.
    /// </summary>
    class Activation
    {
        /// <summary>
        /// Exponential Linear Unit (AKA Scaled ELU or SELU)
        /// Returns an exponential function when zero or lower.
        /// For multiplier = 1, the function is continuously differentiable at all points.
        /// </summary>
        /// <param name="val"></param>
        /// <param name="multiplier"></param>
        /// <returns></returns>
        public static double ELU(double val, double multiplier)
        {
            if(val > 0)
            {
                return val;
            }
            else
            {
                return multiplier * (Math.Exp(val) - 1);
            }
        }

        /// <summary>
        /// Parametric ReLU with a constant of 0.01.
        /// </summary>
        /// <param name="val">Node value.</param>
        /// <returns>If val is greater than zero then val. If val is less than or equal to zero then 0.01 times val.</returns>
        public static double LeakyReLU(double val)
        {
            return PReLU(val, 0.01);
        }

        /// <summary>
        /// Parametric ReLU.
        /// ReLU except instead of rectifying to zero, rectify to a multiple of the value.
        /// </summary>
        /// <param name="val">Node Value.</param>
        /// <param name="multiplier">a</param>
        /// <returns>The value if it's greater than 0. The value multiplied by a constant otherwise.</returns>
        public static double PReLU(double val, double multiplier)
        {
            if(val > 0)
            {
                return val;
            }
            else
            {
                return multiplier * val;
            }
        }

        /// <summary>
        /// Application of the Rectified Linear Unit activation function.
        /// </summary>
        /// <param name="val">Node value.</param>
        /// <returns>Node value with negative values rectified to zero.</returns>
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

        /// <summary>
        /// Application of the "sigmoid," or logistic, mathematical function.
        /// </summary>
        /// <param name="val">Node value.</param>
        /// <param name="derivative">Performs the derivative of the sigmoid function if true.</param>
        /// <returns>The result of either the sigmoid function or its derivative.</returns>
        /// <remarks>
        /// The derivative of the sigmoid function can be expressed with the sigmoid function itself.
        /// </remarks>
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

        /// <summary>
        /// SoftMax formula implementation.
        /// Since the result of each value is dependant upon the rest of the values in that column vector,
        /// the whole matrix of node values is required to compute each value.
        /// </summary>
        /// <param name="values">Matrix of node values from a layer.</param>
        /// <returns>New matrix of node values.</returns>
        public static Matrix<float> SoftMax(Matrix<float> values)
        {
            //  Modify the values matrix for data integrity.
            Matrix<float> valuesModified = values.Clone();
            foreach(Vector<float> valueColumn in valuesModified.EnumerateColumns())
            {
                //  Find the maximum value in each column vector
                float max = float.MinValue;
                foreach(float value in valueColumn)
                {
                    max = Math.Max(value, max);
                }

                //  Subtract the max value of each column vector from all the elements of that vector.
                valueColumn.MapInplace((float n) =>
                {
                    return n - max;
                });
            }

            Matrix<float> exps = Matrix<float>.Build.Dense(valuesModified.RowCount, valuesModified.ColumnCount);
            valuesModified.Map((float a) =>
            {
                return (float)Math.Exp(a);
            },
            exps);

            Vector<float> sums = exps.ColumnSums();

            return exps.MapIndexed((int row, int column, float a) =>
            {
                return a / sums.At(column);
            });
        }

        public static Matrix<float> SoftMaxDerivative(Matrix<float> values, Matrix<float> target, Func<float,float,float> costDerivative)
        {
            Matrix<float> soft = SoftMax(values);

            Matrix<float> outputgradient = soft.MapIndexed((int row, int column, float n) =>
            {
                return costDerivative(n, target.At(row, column));
            });

            Matrix<float> deltas = Matrix<float>.Build.SameAs(values);

            for(int i = 0; i < values.ColumnCount; i++)
            {
                Matrix<float> jacobian = SoftMaxJacobian(values.Column(i), soft.Column(i));
                Vector<float> columnDelta = jacobian.Multiply(outputgradient.Column(i));

                deltas.SetColumn(i,columnDelta);
            }

            return deltas;
        }

        public static Matrix<float> SoftMaxJacobian(Vector<float> values, Vector<float> soft)
        {
            //  Populate jacobian matrix using the formulas for the derivatives of softmax
            Matrix<float> Jacobian = Matrix<float>.Build.Dense(values.Count, values.Count, (int row, int column) =>
             {
                 // Kronecker... your delta is weird.
                 if(row == column)
                 {
                     return soft.At(column) * (1 - soft.At(row));
                 }
                 else
                 {
                     return soft.At(column) * (0 - soft.At(row));
                 }
             });

            return Jacobian;
        }

        /// <summary>
        /// A simple step function with a discontinuity at zero.
        /// </summary>
        /// <param name="val">Node value.</param>
        /// <returns>One if the input is greater than zero and zero otherwise.</returns>
        public static double Step(double val)
        {
            if(val <= 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        /// <summary>
        /// Application of the Hyperbolic Tangent function.
        /// </summary>
        /// <param name="var">Node Value.</param>
        /// <returns>Result of the hyperbolic function.</returns>
        public static double TanH(double var)
        {
            return (2 / (1 + Math.Exp(-var))) - 1;
        }
    }
}
