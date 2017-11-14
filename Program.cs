namespace Classifier
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows;
    using System.IO;

    class Program
    {
        public static int NumLayers = 4;
        public static int InputSize = 2;
        public static int[] LayerSize = {10, 10, 10, 1};
        public const double Alpha = 0.001;

        public static double DesiredPrecision = 1.0;
        public static string FilePath = @"C:\Users\Bishoy\Google Drive\Fall 2016\CSCE 5262 - Neural Networks and Genetic Algorithms\4. Assignments\01\files\spirals";
        public static int ImageDimension = 512;
        public static int Step = 100;

        static void Main(string[] args)
        {
            Net N = new Net(NumLayers, InputSize, LayerSize);
            var Points = ParseTrainingPoints(FilePath + ".txt");
            N.trainForPrecision(Points, DesiredPrecision, FilePath + "_precision.txt", FilePath + "_error.txt", Step);
            N.PrintImage(FilePath + "_boundary.png", 512, Points.Where(x => x.Item2 > 0.0).Select(x => Tuple.Create(x.Item1.X, x.Item1.Y)).ToList());
        }

        public static List<Tuple<Point, double>> ParseTrainingPoints(string filePath)
        {
            List<Tuple<Point, double>> ans = new List<Tuple<Point, double>>();
            StreamReader reader = new StreamReader(filePath);
            for (string line = reader.ReadLine(); line != null; line = reader.ReadLine())
            {
                string[] arr = line.Split('\t');
                ans.Add(Tuple.Create(new Point(double.Parse(arr[0]), double.Parse(arr[1])), int.Parse(arr[2]) == 1 ? 1.0 : -1.0));
            }

            return ans;
        }
    }
}
