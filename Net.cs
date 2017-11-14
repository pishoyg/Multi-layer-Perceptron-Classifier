namespace Classifier
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Drawing;
    using System.Windows;

    class Net
    {
        List<Layer> Layers;             // network layers
        private int NumLayers;          // number of layers
        private int InputSize;          // size of input

        // net constructor
        public Net(int numLayers, int inputSize, int[] hiddenLayerSize)
        {
            InputSize = inputSize;
            NumLayers = numLayers;
            Layers = new List<Layer>();
            Layers.Add(new Layer(InputSize, hiddenLayerSize[0]));
            for (int i = 1; i < NumLayers; ++i)
                Layers.Add(new Layer(hiddenLayerSize[i - 1], hiddenLayerSize[i]));
        }

        public void input(double[] x)
        {
            Layers[0].input(x);
            for (int i = 1; i < NumLayers; ++i)
                Layers[i].input(Layers[i - 1].Output);
        }

        public double [] Output
        {
            set { }
            get
            {
                return Layers[Layers.Count - 1].Output;
            }
        }

        public void trainOnSample(double[] input, double[] correctOutput)
        {
            // give this sample to the network to compute
            this.input(input);

            // set the deltas
            Layers[Layers.Count - 1].SetOutputLayerDelta(correctOutput);
            for (int i = Layers.Count - 2; i >= 0; --i)
            {
                Layers[i].SetHiddenLayerDelta(Layers[i + 1]);
            }

            // now, adjust the weights
            Layers[0].adjustWeights(input, Program.Alpha);
            for (int i = 1; i < NumLayers; ++i)
                Layers[i].adjustWeights(Layers[i - 1].Output, Program.Alpha);
        }

        public Tuple<double, double> trainOnEpoch(List<Tuple<System.Windows.Point, double>> points)
        {
            /*
            Random R = new Random();
            for (int i = 0; i < points.Count; ++i)
            {
                int j = R.Next() % points.Count;
                var temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
            */

            double Error = 0.0;
            int correctCount = 0;
            foreach(var point in points)
            {
                this.trainOnSample(new double[] {point.Item1.X, point.Item1.Y}, new double[] {point.Item2});
                Error += (this.Output[0] - point.Item2) * (this.Output[0] - point.Item2);

                if ((this.Output[0] > 0.0) == (point.Item2 > 0.0))
                {
                    ++correctCount;
                }
            }

            // precision and error!
            return Tuple.Create((double)correctCount / points.Count, Error * 0.5);
        }

        public void trainForPrecision(List<Tuple<System.Windows.Point, double>> points, double desiredPrecision, string precisionFileName, string errorFileName, int step)
        {
            System.IO.StreamWriter ErrorFile = new System.IO.StreamWriter(errorFileName);
            System.IO.StreamWriter PrecisionFile = new System.IO.StreamWriter(precisionFileName);

            double currentPrecision = 0.0, currentError = 1.0;
            int epochCounter = 0;
            for (; currentPrecision < desiredPrecision; ++epochCounter)
            {
                var currentState = trainOnEpoch(points);
                currentPrecision = currentState.Item1;
                currentError = currentState.Item2;
                if (epochCounter % step == 0)
                {
                    Console.WriteLine("{0}\t: {1}", epochCounter, currentPrecision);
                    ErrorFile.Write("{0}\t", currentError);
                    PrecisionFile.Write("{0}\t", currentPrecision);
                }

                if (epochCounter % 10000 == 0)
                {
                    PrintImage(errorFileName.Substring(0, errorFileName.Length - 10) + epochCounter.ToString() + ".png", 512, points.Select(x => Tuple.Create(x.Item1.X, x.Item1.Y)).ToList());
                }
            }

            ErrorFile.Write("{0}", currentError);
            PrecisionFile.Write("{0}", currentPrecision);
            ErrorFile.Close();
            PrecisionFile.Close();
        }
        /// <summary>
        /// this function prints the current state of the network as an image
        /// </summary>
        /// <param name="imageFilePath">name of file to print results in</param>
        /// <param name="dimension">number of pixels of the picture</param>
        /// <param name="specialPoints">a list of points representing the yellow class, to be highlighted in the diagram</param>
        public void PrintImage(string imageFilePath, int dimension, List<Tuple<double, double>> specialPoints = null)
        {
            int[] dx = new int[]{1, 1, 1, 0, 0, 0, -1, -1, -1};
            int[] dy = new int[]{1, 0, -1, 1, 0, -1, 1, 0, -1};
            int Origin = dimension >> 1;
            // origin would be (dimension, dimension)
            Bitmap image = new Bitmap(dimension, dimension);
            Random R = new Random();
            for (int i = 0; i < dimension; ++i)
            {
                for (int j = 0; j < dimension; ++j)
                {
                    double[] coordinates = { (double)(i - Origin) / Origin, (double)(j - Origin) / Origin };
                    this.input(coordinates);
                    image.SetPixel(i, j, this.Output[0] > 0.0 ? Color.Yellow : Color.Blue);
                }
            }

            if (specialPoints != null)
            {
                foreach (Tuple<double, double> point in specialPoints)
                {
                    int x = (int)(point.Item1 * Origin) + Origin, y = (int)(point.Item2 * Origin) + Origin;
                    for (int i = 0; i < 9; ++i)
                    {
                        int cur_x = x + dx[i], cur_y = y + dy[i];
                        if (cur_x >= 0 && cur_x < dimension && cur_y >= 0 && cur_y < dimension)
                        image.SetPixel(cur_x, cur_y, Color.Black);
                    }
                }
            }

            image.Save(imageFilePath);
        }
    }
}
