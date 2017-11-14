namespace Classifier
{
    using System;

    class BipolarSigmoid
    {
        private const double Sigma = 1.0;

        // passed x, return f(x)
        public static double F(double value)
        {
            // return 1.0 / (1.0 + Math.Exp(-value));
            return (2.0 / (1.0 + Math.Exp(-Sigma * value))) - 1.0;
        }

        // passed the value of f(x), return f'(x)
        public static double D(double value)
        {
            // return value * (1.0 - value);
            return (1.0 + value) * (1.0 - value) * Sigma * 0.5;
        }
    }
}
