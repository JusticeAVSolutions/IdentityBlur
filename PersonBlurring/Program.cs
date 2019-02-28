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
            CascadeClassifier cascade = new CascadeClassifier(@"C:\Users\MatthewM\Documents\IdentityBlur\PersonBlurring\haarcascade_frontalface_alt.xml");
            Console.WriteLine("Haar Loaded");
            VideoCapture captureInstance = new VideoCapture(0);
            while (!captureInstance.IsOpened())
            {
                Console.WriteLine("Video Capture being reopened.");
                captureInstance.Open(0);
                Thread.Sleep(500);
            }
            Console.WriteLine("Camera Opened");
            Mat frame = new Mat();
            Mat gray = new Mat();
            while (!cts.IsCancellationRequested)
            {
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
                    Mat roi = new Mat(frame, new Rect((rect.X - rect.Width * .1 < 0) ? 0 : (int)(rect.X - rect.Width*.1),
                                                      (rect.Y - rect.Height * .2 < 0) ? 0 : (int)(rect.Y - rect.Height * .2),
                                                      (frame.Width < rect.Width*1.1 + rect.X) ? frame.Width-rect.X + (int)(rect.Width*.1) : (int)(rect.Width*1.2),
                                                      (frame.Height < rect.Height*1.2 + rect.Y) ? frame.Height-rect.Y + (int)(rect.Height * .2) : (int)(rect.Height * 1.4)));
                    Cv2.GaussianBlur(roi, roi, new Size(101, 101), 0);
                    Cv2.Rectangle(frame, rect, new Scalar(255, 255, 255), 1);
                }
                Cv2.ImShow("Feed", frame);
                Cv2.WaitKey(1);
            }
        }
    }
}
