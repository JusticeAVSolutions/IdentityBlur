using System;
using System.Threading;
using OpenCvSharp;

namespace PersonBlurring
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Started");
            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (sender, a) =>
            {
                a.Cancel = true;
                cts.Cancel();
            };
            CascadeClassifier cascade = new CascadeClassifier(@"C:\Users\MatthewM\Documents\PersonBlurring\PersonBlurring\haarcascade_frontalface_alt.xml");
            Console.WriteLine("Haar Loaded");
            VideoCapture captureInstance = new VideoCapture(0);
            while (!captureInstance.IsOpened())
            {
                Console.WriteLine("Video Capture being reopened.");
                captureInstance.Open(0);
                Thread.Sleep(500);
            }
            Console.WriteLine("Camera Opened");
            while (!cts.IsCancellationRequested)
            {
                Mat frame = new Mat();
                Mat gray = new Mat();
                captureInstance.Read(frame);
                if (frame.Empty())
                    break;
                Cv2.CvtColor(frame, gray, ColorConversionCodes.BGRA2GRAY);
                var faces = cascade.DetectMultiScale(       //detects faces and stores locations
                    image: gray,
                    scaleFactor: 1.1,
                    minNeighbors: 3,
                    flags: (HaarDetectionType)0,
                    minSize: null
                    );
                foreach (Rect rect in faces)
                {
                    Cv2.Rectangle(frame, rect, new Scalar(255, 255, 255), -1);
                }
                Cv2.ImShow("Feed", frame);
                Cv2.WaitKey(1);
            }
        }
    }
}
