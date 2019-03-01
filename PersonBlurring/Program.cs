using System;
using System.Threading;
using System.Drawing;
using OpenCvSharp;
using DlibDotNet;

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
            CorrelationTracker tracker = new CorrelationTracker();
            Mat frame = new Mat();
            Mat gray = new Mat();
            bool trackingFace = false;
            while (!cts.IsCancellationRequested)
            {
                captureInstance.Read(frame);
                if (frame.Empty())
                    break;
                if (!trackingFace)
                {
                    Cv2.CvtColor(frame, gray, ColorConversionCodes.BGRA2GRAY);
                    var faces = cascade.DetectMultiScale(
                        image: gray,
                        scaleFactor: 1.1,
                        minNeighbors: 3,
                        flags: (HaarDetectionType)0,
                        minSize: null
                        );
                    if (faces.Length > 0)
                    {
                        MatToBitmap(frame).Save(@"C:\Users\MatthewM\Documents\IdentityBlur\PersonBlurring\temp.bmp", System.Drawing.Imaging.ImageFormat.Bmp);
                        tracker.StartTrack(DlibDotNet.Dlib.LoadImage<int>(@"C:\Users\MatthewM\Documents\IdentityBlur\PersonBlurring\temp.bmp"), new DRectangle(faces[0].X - 10, faces[0].Y - 10, faces[0].X + faces[0].Width + 10, faces[0].Y + faces[0].Height + 20));
                        trackingFace = true;
                    }
                }
                else
                {
                    MatToBitmap(frame).Save(@"C:\Users\MatthewM\Documents\IdentityBlur\PersonBlurring\temp.bmp", System.Drawing.Imaging.ImageFormat.Bmp);
                    double trackingQuality = tracker.Update(DlibDotNet.Dlib.LoadImage<int>(@"C:\Users\MatthewM\Documents\IdentityBlur\PersonBlurring\temp.bmp"));
                    if (trackingQuality >= 5)
                    {
                        DRectangle position = tracker.GetPosition();
                        Mat roi = new Mat(frame, new Rect((position.TopLeft.X < 0) ? 0 : (int)position.TopLeft.X,
                                                          (position.TopLeft.Y < 0) ? 0 : (int)position.TopLeft.Y,
                                                          (frame.Width < position.Width + position.TopLeft.X) ? frame.Width - (int)position.TopLeft.X : (int)position.Width,
                                                          (frame.Height < position.Height + position.TopLeft.Y) ? frame.Height - (int)position.TopLeft.Y : (int)position.Height));
                        Cv2.GaussianBlur(roi, roi, new OpenCvSharp.Size(101, 101), 0);
                        Cv2.Rectangle(frame, new Rect((int)position.TopLeft.X, (int)position.TopLeft.Y, (int)position.Width, (int)position.Height), new Scalar(255, 255, 255));
                    }
                    else
                    {
                        trackingFace = false;
                    }
                }
                /*foreach (Rect rect in faces)
                {
                    Mat roi = new Mat(frame, new Rect((rect.X - rect.Width * .1 < 0) ? 0 : (int)(rect.X - rect.Width*.1),
                                                      (rect.Y - rect.Height * .2 < 0) ? 0 : (int)(rect.Y - rect.Height * .2),
                                                      (frame.Width < rect.Width*1.1 + rect.X) ? frame.Width-rect.X + (int)(rect.Width*.1) : (int)(rect.Width*1.2),
                                                      (frame.Height < rect.Height*1.2 + rect.Y) ? frame.Height-rect.Y + (int)(rect.Height * .2) : (int)(rect.Height * 1.4)));
                    Cv2.GaussianBlur(roi, roi, new Size(101, 101), 0);
                    Cv2.Rectangle(frame, rect, new Scalar(255, 255, 255), 1);
                }*/
                Cv2.ImShow("Feed", frame);
                Cv2.WaitKey(1);
            }
        }
        private static Bitmap MatToBitmap(Mat mat)
        {
            using (var ms = mat.ToMemoryStream())
            {
                return (Bitmap)Image.FromStream(ms);
            }
        }
    }
}
