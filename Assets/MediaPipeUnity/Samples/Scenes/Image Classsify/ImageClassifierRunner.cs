using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe.Tasks.Vision.ImageClassifier;
using UnityEngine.Rendering;
using ImageClassifierResult = Mediapipe.Tasks.Components.Containers.ClassificationResult;
using Unity.Collections;
using System.Text;
using System.IO;
using System.Diagnostics;
using Mediapipe.Unity.Sample.FaceDetection;
using FaceDetectionResult = Mediapipe.Tasks.Components.Containers.DetectionResult;
using Mediapipe.Unity.Experimental;

using Unity.Collections.LowLevel.Unsafe;
using Mediapipe;
using System;
using Mediapipe.Tasks.Components.Containers;
using System.Drawing;
using UnityEngine.UI;
namespace Mediapipe.Unity.Sample.ImageClassification
{
    public class ImageClassifierRunner : VisionTaskApiRunner<ImageClassifier>
    {
        //[SerializeField] private ImageClassifierResultAnnotationController _imageClassifierResultAnnotationController;

        private Experimental.TextureFramePool _textureFramePool;

        public readonly ImageClassificationConfig config = new ImageClassificationConfig();

        private Tasks.Vision.FaceDetector.FaceDetector faceDetector;
        public readonly FaceDetectionConfig faceDetectionConfig = new FaceDetectionConfig();
        public class ImagePreprocessing
        {
            public static Texture2D CenterCrop(Texture2D texture, int size)//没问题
            {
                int width = texture.width;
                int height = texture.height;
                int cropSize = Mathf.Min(width, height);
                int top = (height - cropSize) / 2;
                int left = (width - cropSize) / 2;

                Texture2D croppedTexture = new Texture2D(cropSize, cropSize);
                UnityEngine.Color[] pixels = texture.GetPixels(left, top, cropSize, cropSize);
                croppedTexture.SetPixels(pixels);
                croppedTexture.Apply();
                //SaveTextureAsPNG(croppedTexture, "C:/Users/qxmz/Desktop/resizedTexture3_qian.png");//正常的

                //Texture2D resizedTexture = new Texture2D(size, size);
                //Graphics.ConvertTexture(croppedTexture, resizedTexture);//这个就不对，ConvertTexture用法不对
                //SaveTextureAsPNG(resizedTexture, "C:/Users/qxmz/Desktop/resizedTexture2_hou.png");//不正常，Convert用法不对

                Texture2D resizedTexture = ResizeTexture(croppedTexture, 224, 224);
                //SaveTextureAsPNG(resizedTexture, "C:/Users/qxmz/Desktop/resizedTexture3_hou.png");
                return resizedTexture;
            }

            private static Texture2D ResizeTexture(Texture2D source, int newWidth, int newHeight)
            {
                // 创建一个RenderTexture并设置大小和格式
                RenderTexture rt = new RenderTexture(newWidth, newHeight, 24);
                rt.filterMode = FilterMode.Bilinear;

                // 将source贴图设置为活动的RenderTexture
                RenderTexture.active = rt;

                // 使用Graphics.Blit将source贴图渲染到RenderTexture上
                Graphics.Blit(source, rt);

                // 创建一个新的Texture2D并读取RenderTexture中的像素
                Texture2D result = new Texture2D(newWidth, newHeight);
                result.ReadPixels(new UnityEngine.Rect(0, 0, newWidth, newHeight), 0, 0);
                result.Apply();

                // 重置活动的RenderTexture
                RenderTexture.active = null;

                // 释放RenderTexture
                rt.Release();

                return result;
            }

            private static void SaveTextureAsPNG(Texture2D texture, string filePath)
            {
                // 将Texture2D编码为PNG
                byte[] bytes = texture.EncodeToPNG();
                // 将PNG字节数组写入文件
                File.WriteAllBytes(filePath, bytes);
                UnityEngine.Debug.Log($"Texture saved as {filePath}");
            }

            //public static float[] Normalize(float[] tensor, float[] mean, float[] std)
            //{
            //    for (int i = 0; i < tensor.Length; i++)
            //    {
            //        tensor[i] = (tensor[i] - mean[i % mean.Length]) / std[i % std.Length];
            //    }
            //    return tensor;
            //}

            public static float[] Normalize(float[] tensor, float[] mean, float[] std)
            {
                for (int i = 0; i < tensor.Length / 3; i++)
                {
                    tensor[i * 3 + 0] = (tensor[i * 3 + 0] - mean[0]) / std[0];
                    tensor[i * 3 + 1] = (tensor[i * 3 + 1] - mean[1]) / std[1];
                    tensor[i * 3 + 2] = (tensor[i * 3 + 2] - mean[2]) / std[2];
                }
                return tensor;
            }

            public static float[] PreprocessTexture(Texture2D texture)
            {

                UnityEngine.Color[] originalPixels = texture.GetPixels();


                // CenterCrop to 224x224 (or the required size)
                Texture2D croppedResizedTexture = CenterCrop(texture, 224);

                // Convert to float array and normalize
                UnityEngine.Color[] pixels = croppedResizedTexture.GetPixels();
                float[] tensor = new float[pixels.Length * 3];

                //不需要归一化，pixels已经自动归一化过了
                //for (int i = 0; i < pixels.Length; i++)
                //{
                //    tensor[i * 3 + 0] = pixels[i].r / 255.0f;
                //    tensor[i * 3 + 1] = pixels[i].g / 255.0f;
                //    tensor[i * 3 + 2] = pixels[i].b / 255.0f;
                //}

                for (int i = 0; i < pixels.Length; i++)
                {
                    tensor[i * 3 + 0] = pixels[i].r;
                    tensor[i * 3 + 1] = pixels[i].g;
                    tensor[i * 3 + 2] = pixels[i].b;
                }



                // Normalize with mean and std
                float[] mean = { 0.485f, 0.456f, 0.406f };
                float[] std = { 0.229f, 0.224f, 0.225f };
                tensor = Normalize(tensor, mean, std);

                return tensor;
            }

            public static Texture2D PreprocessTexture_onlyCenterCrop(Texture2D texture)
            {
                // CenterCrop to 224x224 (or the required size)
                Texture2D croppedResizedTexture = CenterCrop(texture, 224);


                return croppedResizedTexture;
            }
        }

        public UnityEngine.UI.Image preview;

        public Image DuplicateImage(Image originalImage)
        {
            // 将原始图像转换为CPU图像以确保像素数据在CPU内存中
            originalImage.ConvertToCpu();

            // 获取图像的属性
            int width = originalImage.Width();
            int height = originalImage.Height();
            int channels = originalImage.Channels();
            int step = originalImage.Step();
            var format = originalImage.ImageFormat();

            // 创建一个用于存储像素数据的NativeArray
            int pixelDataSize = height * step;
            NativeArray<byte> pixelData = new NativeArray<byte>(pixelDataSize, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            // 锁定像素数据并复制到NativeArray
            using (var pixelWriteLock = new PixelWriteLock(originalImage))
            {
                IntPtr pixelsPtr = pixelWriteLock.Pixels();
                unsafe
                {
                    byte* sourcePtr = (byte*)pixelsPtr.ToPointer();
                    byte* destPtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(pixelData);

                    for (int i = 0; i < pixelDataSize; i++)
                    {
                        destPtr[i] = sourcePtr[i];
                    }
                }
            }

            // 创建一个新的图像对象
            var newImage = new Image(format, width, height, step, pixelData);

            // 释放NativeArray
            pixelData.Dispose();

            return newImage;
        }

        private static Image Rotate90CounterClockwise(Image image)//逆时针90°
        {
            image.ConvertToCpu();
            // 获取原始图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();
            
            var format = image.ImageFormat();

            int newWidth = height;
            int newHeight = width;
            int widthStep = channels * newWidth;

            NativeArray<byte> pixelData = new NativeArray<byte>(newWidth * newHeight * channels, Allocator.Temp);
            Image newImage = new Image(format, newWidth, newHeight, widthStep, pixelData);


            using (var inputLock = new PixelWriteLock(image))
            using (var outputLock = new PixelWriteLock(newImage))
            {
                IntPtr inputPixelsPtr = inputLock.Pixels();
                IntPtr outputPixelsPtr = outputLock.Pixels();

                unsafe
                {
                    byte* inputPixels = (byte*)inputPixelsPtr.ToPointer();
                    byte* outputPixels = (byte*)outputPixelsPtr.ToPointer();

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                int inputIndex = (y * width + x) * channels + c;
                                int outputIndex = ((newWidth - 1 - y) + x * newWidth) * channels + c;
                                outputPixels[outputIndex] = inputPixels[inputIndex];
                            }
                        }
                    }
                }
            }

            return newImage ;
        }

        private static Image Rotate90Clockwise(Image image)//顺时针90°
        {
            image.ConvertToCpu();
            // 获取原始图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();

            var format = image.ImageFormat();

            int newWidth = height;
            int newHeight = width;
            int widthStep = channels * newWidth;

            NativeArray<byte> pixelData = new NativeArray<byte>(newWidth * newHeight * channels, Allocator.Temp);
            Image newImage = new Image(format, newWidth, newHeight, widthStep, pixelData);

            using (var inputLock = new PixelWriteLock(image))
            using (var outputLock = new PixelWriteLock(newImage))
            {
                IntPtr inputPixelsPtr = inputLock.Pixels();
                IntPtr outputPixelsPtr = outputLock.Pixels();

                unsafe
                {
                    byte* inputPixels = (byte*)inputPixelsPtr.ToPointer();
                    byte* outputPixels = (byte*)outputPixelsPtr.ToPointer();

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                int inputIndex = (y * width + x) * channels + c;
                                int outputIndex = (y + (newHeight - 1 - x) * newWidth) * channels + c;
                                outputPixels[outputIndex] = inputPixels[inputIndex];
                            }
                        }
                    }
                }
            }

            return newImage;
        }
        private static Image CropImage(Image image, Mediapipe.Tasks.Components.Containers.Rect boundingBox)
        {
            // 将原始图像转换为CPU图像以确保像素数据在CPU内存中
            image.ConvertToCpu();

            // 获取原始图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();
            int step = image.Step();
            var format = image.ImageFormat();


            // 获取裁剪区域的坐标和大小 for PC
            int x = boundingBox.left;
            int y = boundingBox.top;
            int cropWidth = boundingBox.right - boundingBox.left;
            int cropHeight = boundingBox.bottom - boundingBox.top;

           

            // 确保裁剪区域在图像内
            //if (x < 0 || y < 0 || cropWidth <= 0 || cropHeight <= 0 || x + cropWidth > width || y + cropHeight > height)
            //{
            //    throw new ArgumentException("Invalid bounding box dimensions.");
            //}

            // 创建一个用于存储裁剪后像素数据的NativeArray
            int cropDataSize = cropHeight * step;
            NativeArray<byte> cropPixelData = new NativeArray<byte>(cropDataSize, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            // 锁定原图像的像素数据并进行裁剪
            using (var pixelWriteLock = new PixelWriteLock(image))
            {
                IntPtr pixelsPtr = pixelWriteLock.Pixels();
                unsafe
                {
                    byte* sourcePtr = (byte*)pixelsPtr.ToPointer();
                    byte* destPtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(cropPixelData);

                    // 计算每行的字节步进
                    int srcStep = image.Step();

                    // 执行裁剪操作
                    for (int row = 0; row < cropHeight; row++)
                    {
                        byte* srcRow = sourcePtr + (y + row) * srcStep + x * channels;
                        byte* destRow = destPtr + row * cropWidth * channels;

                        for (int col = 0; col < cropWidth; col++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                destRow[col * channels + c] = srcRow[col * channels + c];
                            }
                        }
                    }
                }
            }

            // 创建一个新的图像对象
            var newImage = new Image(format, cropWidth, cropHeight, cropWidth * channels, cropPixelData);

            // 释放NativeArray
            cropPixelData.Dispose();

            return newImage;
        }

        private static Image CropImage1(Image image, Mediapipe.Tasks.Components.Containers.Rect boundingBox)
        {
            // 将原始图像转换为CPU图像以确保像素数据在CPU内存中
            image.ConvertToCpu();

            // 获取原始图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();
            int step = image.Step();
            var format = image.ImageFormat();


            // 获取裁剪区域的坐标和大小 for mobile
            //int x = boundingBox.left;
            int x = boundingBox.top;
            //int y = boundingBox.top;
            int y = boundingBox.right;

            //int cropWidth = boundingBox.right - boundingBox.left;
            int cropHeight = boundingBox.right - boundingBox.left;

            //int cropHeight = boundingBox.bottom - boundingBox.top;
            int cropWidth = boundingBox.bottom - boundingBox.top;



            // 确保裁剪区域在图像内
            //if (x < 0 || y < 0 || cropWidth <= 0 || cropHeight <= 0 || x + cropWidth > width || y + cropHeight > height)
            //{
            //    throw new ArgumentException("Invalid bounding box dimensions.");
            //}

            // 创建一个用于存储裁剪后像素数据的NativeArray
            //int cropDataSize = cropHeight * step;//step要改
            int cropDataSize = cropHeight * cropWidth * channels;
            NativeArray<byte> cropPixelData = new NativeArray<byte>(cropDataSize, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            // 锁定原图像的像素数据并进行裁剪
            using (var pixelWriteLock = new PixelWriteLock(image))
            {
                IntPtr pixelsPtr = pixelWriteLock.Pixels();
                unsafe
                {
                    byte* sourcePtr = (byte*)pixelsPtr.ToPointer();//image左上角
                    byte* destPtr = (byte*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(cropPixelData);//也是左上角

                    // 计算每行的字节步进
                    //int srcStep = image.Step();//srcstep要改
                    int srcStep = cropWidth * channels;
                    // 执行裁剪操作
                    for (int row = 0; row < cropHeight; row++)
                    {
                        byte* srcRow = sourcePtr + (y + row) * srcStep + x * channels;
                        byte* destRow = destPtr + row * cropWidth * channels;

                        for (int col = 0; col < cropWidth; col++)
                        {
                            for (int c = 0; c < channels; c++)
                            {
                                destRow[col * channels + c] = srcRow[col * channels + c];
                            }
                        }
                    }
                }
            }

            // 创建一个新的图像对象
            var newImage = new Image(format, cropWidth, cropHeight, cropWidth * channels, cropPixelData);

            // 释放NativeArray
            cropPixelData.Dispose();

            return newImage;
        }


        private static void ShowTextureInConsole(Texture2D texture)
        {
            GameObject canvas = new GameObject("Canvas");
            Canvas c = canvas.AddComponent<Canvas>();
            c.renderMode = RenderMode.ScreenSpaceOverlay;
            CanvasScaler cs = canvas.AddComponent<CanvasScaler>();
            cs.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvas.AddComponent<GraphicRaycaster>();

            GameObject imageObject = new GameObject("Image");
            imageObject.transform.SetParent(canvas.transform);
            UnityEngine.UI.Image img = imageObject.AddComponent< UnityEngine.UI.Image>();
            RectTransform rt = imageObject.GetComponent<RectTransform>();
            rt.sizeDelta = new Vector2(texture.width, texture.height);

            Sprite sprite = Sprite.Create(texture, new UnityEngine.Rect(0, 0, texture.width, texture.height), new Vector2(0.5f, 0.5f));
            img.sprite = sprite;
        }
        private static void SaveImageAsPNG(Image image, string filePath)
        {
            // 将Image转换为CPU图像以确保像素数据在CPU内存中
            image.ConvertToCpu();

            // 获取图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();

            // 创建一个Texture2D对象
            Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);

            // 锁定图像的像素数据
            using (var pixelWriteLock = new PixelWriteLock(image))
            {
                IntPtr pixelsPtr = pixelWriteLock.Pixels();
                unsafe
                {
                    byte* sourcePtr = (byte*)pixelsPtr.ToPointer();
                    Color32[] colors = new Color32[width * height];

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int index = y * width + x;
                            int pixelIndex = index * channels;
                            colors[index] = new Color32(
                                sourcePtr[pixelIndex],
                                sourcePtr[pixelIndex + 1],
                                sourcePtr[pixelIndex + 2],
                                channels == 4 ? sourcePtr[pixelIndex + 3] : (byte)255);
                        }
                    }

                    texture.SetPixels32(colors);
                    texture.Apply();
                }
            }

            // 将Texture2D编码为PNG
            byte[] pngData = texture.EncodeToPNG();

            // 将PNG字节数组写入文件
            File.WriteAllBytes(filePath, pngData);
            //ShowTextureInConsole(texture);

        }

        private static void showOnScreen(Image image, UnityEngine.UI.Image preview)
        {
            // 将Image转换为CPU图像以确保像素数据在CPU内存中
            image.ConvertToCpu();

            // 获取图像的属性
            int width = image.Width();
            int height = image.Height();
            int channels = image.Channels();

            // 创建一个Texture2D对象
            Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);

            // 锁定图像的像素数据
            using (var pixelWriteLock = new PixelWriteLock(image))
            {
                IntPtr pixelsPtr = pixelWriteLock.Pixels();
                unsafe
                {
                    byte* sourcePtr = (byte*)pixelsPtr.ToPointer();
                    Color32[] colors = new Color32[width * height];

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int index = y * width + x;
                            int pixelIndex = index * channels;
                            colors[index] = new Color32(
                                sourcePtr[pixelIndex],
                                sourcePtr[pixelIndex + 1],
                                sourcePtr[pixelIndex + 2],
                                channels == 4 ? sourcePtr[pixelIndex + 3] : (byte)255);
                        }
                    }

                    texture.SetPixels32(colors);
                    texture.Apply();
                }
            }

           Sprite sprite = Sprite.Create(texture, new UnityEngine.Rect(0, 0, texture.width, texture.height), Vector2.zero);
           preview.sprite = sprite;  
            

        }
        public Texture2D LoadTextureFromFile(string filePath)
        {
            Texture2D texture = new Texture2D(2, 2);
            byte[] fileData = File.ReadAllBytes(filePath);
            texture.LoadImage(fileData); // Auto-resize the texture based on file data
            return texture;
        }

        public Texture2D Rgb2Rgba(Texture2D texture)
        {
            Texture2D sRGBAOutput = new Texture2D(texture.width, texture.height, TextureFormat.RGBA32, false);
            for (int y = 0; y < texture.height; y++)
            {
                for (int x = 0; x < texture.width; x++)
                {

                    var color = texture.GetPixel(x, y);
                    color.a = 1.0f; // 设置alpha值为1.0
                    sRGBAOutput.SetPixel(x, y, color);
                }
            }

            // 应用像素数据到sRGBAOutput Texture2D
            sRGBAOutput.Apply();
            return sRGBAOutput;

        }

        static void SaveTextureAsPNG1(Texture2D texture, string filePath)
        {
            // 将Texture2D编码为PNG
            byte[] bytes = texture.EncodeToPNG();
            // 将PNG字节数组写入文件
            File.WriteAllBytes(filePath, bytes);
            UnityEngine.Debug.Log($"Texture saved as {filePath}");
        }
        public override void Stop()
        {
            base.Stop();
            _textureFramePool?.Dispose();
            _textureFramePool = null;

        }

        protected override IEnumerator Run()
        {
            UnityEngine.Debug.Log($"Delegate = {config.Delegate}");
            UnityEngine.Debug.Log($"Running Mode = {config.RunningMode}");
            UnityEngine.Debug.Log($"=============================");
            UnityEngine.Debug.Log($"scoreThreshold = {config.scoreThreshold}");

            yield return AssetLoader.PrepareAssetAsync(config.ModelPath);
            yield return AssetLoader.PrepareAssetAsync(faceDetectionConfig.ModelPath); // 新增：准备FaceDetection模型
            //var options = config.GetImageClassifierOptions(config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM ? OnImageClassificationOutput : null);

            // 新增：初始化FaceDetector
            var faceDetectorOptions = faceDetectionConfig.GetFaceDetectorOptions(null);
            faceDetector = Tasks.Vision.FaceDetector.FaceDetector.CreateFromOptions(faceDetectorOptions);

            var options = config.GetImageClassifierOptions(null);
            taskApi = ImageClassifier.CreateFromOptions(options);



            var imageSource = ImageSourceProvider.ImageSource;
            imageSource.SelectSource(0);
            yield return imageSource.Play();

            if (!imageSource.isPrepared)
            {
                UnityEngine.Debug.LogError("Failed to start ImageSource, exiting...");
                yield break;
            }

            _textureFramePool = new Experimental.TextureFramePool(imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10);

            //_textureFramePool = new Experimental.TextureFramePool(224, 224, TextureFormat.RGBA32, 10);
            screen.Initialize(imageSource);

            //SetupAnnotationController(_imageClassifierResultAnnotationController, imageSource);

             var transformationOptions = imageSource.GetTransformationOptions();


            var flipHorizontally = transformationOptions.flipHorizontally;
            var flipVertically = transformationOptions.flipVertically;
            var imageProcessingOptions = new Tasks.Vision.Core.ImageProcessingOptions(rotationDegrees: (int)transformationOptions.rotationAngle);



            AsyncGPUReadbackRequest req = default;
            var waitUntilReqDone = new WaitUntil(() => req.done);
            var result = ImageClassifierResult.Alloc(400);

            // 新增：进行人脸检测
            var faceResult = FaceDetectionResult.Alloc(400);

            int cnt = 0;
            int right_cnt = 0;

            //string folderPath = Application.dataPath + "/MediaPipeUnity/Samples/Scenes/Tasks/Image Classsify/test/Surprise";
            //string[] imagePaths = Directory.GetFiles(folderPath, "*.jpg");

            //foreach (string imagePath in imagePaths)
            //{
            //    cnt++;
            //    Texture2D texture1 = LoadTextureFromFile(imagePath);
            //    //Texture2D texture1 = Resources.Load<Texture2D>("train_Angry0_1");
            //    Texture2D texture = Rgb2Rgba(texture1);
            //    //Texture2D ResizedTexture = ImagePreprocessing.PreprocessTexture_onlyCenterCrop(texture);
            //    //Mediapipe.ImageFormat.Types.Format format = Mediapipe.ImageFormat.Types.Format.Srgb;
            //    Mediapipe.ImageFormat.Types.Format format = Mediapipe.ImageFormat.Types.Format.Srgba;
            //    Mediapipe.Image image2 = new Mediapipe.Image(format, texture);
            //    var image2_90 = Rotate90Clockwise(image2);
            //    var image2_180 = Rotate90Clockwise(image2_90);
            //    //SaveImageAsPNG(image2, "C:/Users/qxmz/Desktop/bendi.png");
            //    //SaveImageAsPNG(image2_180, "C:/Users/qxmz/Desktop/bendi1.png");
            //    taskApi.TryClassify(image2_180, imageProcessingOptions, ref result);
            //    //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].score}***************************************");
            //    //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].categoryName}***************************************");
            //    //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].index}***************************************");
            //    //if (result.classifications[0].categories[0].categoryName == "Fear")
            //    //{
            //    //    right_cnt++;
            //    //}
            //    //UnityEngine.Debug.Log($"{cnt}***************************************");

            //    if (result.classifications.Count > 0 && result.classifications[0].categories.Count > 0)
            //    {
            //        UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].score}***************************************");
            //        //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].categoryName}***************************************");
            //        //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].index}***************************************");
            //        if (result.classifications[0].categories[0].categoryName == "4:Surprise")
            //        {
            //            right_cnt++;
            //        }
            //    }
            //    else
            //    {
            //        UnityEngine.Debug.Log("未检测到分类结果，跳过该图像。");
            //    }
            //}
            //float acc = (float)right_cnt / cnt;
            //string formattedAcc = acc.ToString("F5");
            //UnityEngine.Debug.Log($"Surprise情绪识别准确率:{formattedAcc}***************************************");

            //Application.dataPath只能导航到Assets
            //string ImgPath = Application.dataPath + "/MediaPipeUnity/Samples/Scenes/Tasks/Image Classsify/test_Angry0_1.jpg";
            //Texture2D texture = LoadTextureFromFile(ImgPath);
            ////SaveTextureAsPNG1(texture, "C:/Users/qxmz/Desktop/PreprocessTexture_qian.png");
            ////Texture2D ResizedTexture = ImagePreprocessing.PreprocessTexture_onlyCenterCrop(texture);
            ////SaveTextureAsPNG1(ResizedTexture, "C:/Users/qxmz/Desktop/PreprocessTexture_hou.png");
            //Mediapipe.ImageFormat.Types.Format format = Mediapipe.ImageFormat.Types.Format.Srgb; // 选择合适的格式
            //Mediapipe.Image image2 = new Mediapipe.Image(format, texture);
            //SaveImageAsPNG(image2, "C:/Users/qxmz/Desktop/bendi.png");
            //var image2_90 = Rotate90Clockwise(image2);
            //var image2_180 = Rotate90Clockwise(image2_90);
            //SaveImageAsPNG(image2_180, "C:/Users/qxmz/Desktop/bendi1.png");
            //taskApi.TryClassify(image2, imageProcessingOptions, ref result);
            //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0]}***************************************");
            //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].categoryName}***************************************");


            //时间测试
            float startTime = Time.time;
            int counts = 0;
            float fps = 0;

            while (true)
            {

                

                if (isPaused)
                {
                    yield return new WaitWhile(() => isPaused);
                }
                
                if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))
                {
                    yield return new WaitForEndOfFrame();
                    continue;
                }
                
                req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);
                yield return waitUntilReqDone;
                
                if (req.hasError)
                {
                    UnityEngine.Debug.LogError($"Failed to read texture from the image source, exiting...");
                    break;
                }



                //var image = textureFrame.BuildCPUImage();
                //textureFrame.SaveTextureToLocal();

                var image = textureFrame.BuildCPUImage();
                var image1 = DuplicateImage(image);
                //var normalizedImage = ImageProcessor.NormalizeImage(image);

                switch (taskApi.runningMode)
                {
                    
                    case Tasks.Vision.Core.RunningMode.IMAGE:
                        if (faceDetector.TryDetect(image1, imageProcessingOptions, ref faceResult))
                        {

                            UnityEngine.Debug.Log($"{faceResult.detections[0].boundingBox}=================================");
                            
                            // 扩大矩形框
                            //var originalBoundingBox = faceResult.detections[0].boundingBox;
                            //var expandedBoundingBox = new Mediapipe.Tasks.Components.Containers.Rect(
                            //  left: Math.Max(originalBoundingBox.left - 100, 0),
                            //  top: Math.Max(originalBoundingBox.top - 100, 0),
                            //   right: Math.Min(originalBoundingBox.right + 100, image.Width()),
                            //  bottom: Math.Min(originalBoundingBox.bottom + 100, image.Height())
                            //   );

                            //var image2 = CropImage(image, originalBoundingBox);

                        }
                        var imageToClassify = (faceResult.detections.Count > 0) ? CropImage(image, faceResult.detections[0].boundingBox) : image;

                        //var rotationClassify1 = Rotate90CounterClockwise(imageToClassify);
                        //var rotationClassify1 = Rotate90Clockwise(imageToClassify);
                        //SaveImageAsPNG(imageToClassify, "C:/Users/qxmz/Desktop/bendi2.png");
                        showOnScreen(imageToClassify, preview);
                        //var rotationClassify2 = Rotate90CounterClockwise(rotationClassify1);
                        //var rotationClassify3 = Rotate90CounterClockwise(rotationClassify2);
                        //SaveImageAsPNG(rotationClassify3, "C:/Users/qxmz/Desktop/cropped_image.png");
                        // 裁剪图像
                        //var image2 = CropImage(image, originalBoundingBox);
                        if (taskApi.TryClassify(imageToClassify, imageProcessingOptions, ref result))
                        {
                            UnityEngine.Debug.Log("分类完成");
                            UnityEngine.Debug.Log($"{result.classifications[0]}=================================");
                            //UnityEngine.Debug.Log($"情绪识别:{result.classifications[0].categories[0].categoryName}=================================" );
                            //_imageClassifierResultAnnotationController.DrawNow(result);

                            counts++;
                            float endTime = Time.time;
                            if ((endTime - startTime) > 0)
                            {
                                fps = counts / (endTime - startTime);
                                UnityEngine.Debug.Log($"==============分类完成{fps}");
                                counts = 0;
                                startTime = endTime;
                            }


                        }
                        else
                        {
                            //_imageClassifierResultAnnotationController.DrawNow(default);
                        }
                        break;
                    case Tasks.Vision.Core.RunningMode.VIDEO:
                        if (taskApi.TryClassifyForVideo(image, GetCurrentTimestampMillisec(), imageProcessingOptions, ref result))
                        {
                            //_imageClassifierResultAnnotationController.DrawNow(result);
                            UnityEngine.Debug.Log("分类完成");
                        }
                        else
                        {
                            //_imageClassifierResultAnnotationController.DrawNow(default);
                            UnityEngine.Debug.Log("分类完成");
                        }
                        break;
                    case Tasks.Vision.Core.RunningMode.LIVE_STREAM:
                        taskApi.ClassifyAsync(image, GetCurrentTimestampMillisec(), imageProcessingOptions);
                        break;
                }

                textureFrame.Release();
            }
        }

        private void OnImageClassificationOutput(ImageClassifierResult result, Image image, long timestamp)
        {
            UnityEngine.Debug.Log("进入回调函数");
            //_imageClassifierResultAnnotationController.DrawLater(result);
        }
    }
}
