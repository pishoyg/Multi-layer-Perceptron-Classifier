namespace Classifier
{
    using System;

    /* Notice that in this implementation,
     * each node stores the weights on the edges from the previous layer to herself
     * therefore, to set your output, you only need to be passed the output of the previous layer, which is the input to yourself
     * you have the weights connecting that previous layer to you, and you have everything you need
     */
    class Layer
    {
        public int InputSize;  // number of input neurons
        public int Size;  // number of output neurons

        public double[] YIn;  // input to output neurons {YIn[j] = sum_i(0 <= i < n)x_i*w_ij}
        public double[] Output;    // Y[j] = f(YIn[j])
        public double[] Delta;// Delta
        public double[,] W;   // W[i, j] is the weight on the edge from x_i to y_j
        public double[] Bias;

        /* this constructor sets the sizes of the layer
         * as well as the size of the input to this layer
         */
        public Layer(int inputSize, int size)
        {
            InputSize = inputSize;
            Size = size;
            Bias = new double[Size];
            Output = new double[Size];
            YIn = new double[Size];
            Delta = new double[Size];
            W = new double[InputSize, Size];

            // initialize weights to random numbers
            Random R = new Random();
            for (int i = 0; i < InputSize; ++i)
                for (int j = 0; j < Size; ++j)
                    W[i, j] = (R.NextDouble() - 0.5) / 5.0;
            for (int j = 0; j < Size; ++j) Bias[j] = (R.NextDouble() - 0.5) / 5.0;
        }

        // set the deltas of the neurons, knowing that this is the output layer
        public void SetOutputLayerDelta(double[] correctOutput)
        {
            for (int k = 0; k < this.Size; ++k)
            {
                this.Delta[k] = (correctOutput[k] - this.Output[k]) * BipolarSigmoid.D(this.Output[k]);
            }
        }

        // set the deltas of the neurons, knowing that this is a hidden layer
        public void SetHiddenLayerDelta(Layer next)
        {
            for (int j = 0; j < this.Size; ++j)
            {
                this.Delta[j] = 0.0;
                for (int k = 0; k < next.Size; ++k)
                {
                    this.Delta[j] += (next.W[j, k] * next.Delta[k]);
                }

                this.Delta[j] *= BipolarSigmoid.D(this.Output[j]);
            }
        }

        // passed the input to this layer, and a learning rate, adjust the weights
        // this method assumes that the array Delta has already been filled with proper values!
        public void adjustWeights(double[] previousLayerActivation, double alpha)
        {
            for (int i = 0; i < this.InputSize; ++i)
            {
                for (int j = 0; j < this.Size; ++j)
                {
                    this.W[i, j] += (alpha * this.Delta[j] * previousLayerActivation[i]);
                }
            }
            for (int j = 0; j < this.Size; ++j)
            {
                this.Bias[j] += (alpha * this.Delta[j]);
            }
        }

        /* this method accepts input to the layer,
         * and sets correct values to the output
         *
         * it assumes values of W have already been set
         * but it does not set delta
         */
        public void input(double[] previousLayerActivation)
        {
            // for each neuron in the layer
            for (int j = 0; j < this.Size; ++j)
            {
                // calculate input to the neuron
                this.YIn[j] = this.Bias[j];
                for (int i = 0; i < this.InputSize; ++i)
                {
                    this.YIn[j] += previousLayerActivation[i] * this.W[i, j];
                }
                // calculate output of the neuron
                this.Output[j] = BipolarSigmoid.F(this.YIn[j]);
            }
        }
    }
}
