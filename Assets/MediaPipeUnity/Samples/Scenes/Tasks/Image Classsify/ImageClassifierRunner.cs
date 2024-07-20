using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe.Tasks.Vision.ImageClassifier;
using UnityEngine.Rendering;
using ImageClassifierResult = Mediapipe.Tasks.Components.Containers.ClassificationResult;

namespace Mediapipe.Unity.Sample.ImageClassification
{
    public class ImageClassifierRunner : VisionTaskApiRunner<ImageClassifier>
    {
        //[SerializeField] private ImageClassifierResultAnnotationController _imageClassifierResultAnnotationController;

        private Experimental.TextureFramePool _textureFramePool;

        public readonly ImageClassificationConfig config = new ImageClassificationConfig();

        public override void Stop()
        {
            base.Stop();
            _textureFramePool?.Dispose();
            _textureFramePool = null;
        }

        protected override IEnumerator Run()
        {
            Debug.Log($"Delegate = {config.Delegate}");
            Debug.Log($"Running Mode = {config.RunningMode}");
            Debug.Log($"=============================");
            Debug.Log($"scoreThreshold = {config.scoreThreshold}");

            yield return AssetLoader.PrepareAssetAsync(config.ModelPath);

            //var options = config.GetImageClassifierOptions(config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM ? OnImageClassificationOutput : null);
            var options = config.GetImageClassifierOptions(null);
            

            taskApi = ImageClassifier.CreateFromOptions(options);

            var imageSource = ImageSourceProvider.ImageSource;
            yield return imageSource.Play();

            if (!imageSource.isPrepared)
            {
                Debug.LogError("Failed to start ImageSource, exiting...");
                yield break;
            }

            _textureFramePool = new Experimental.TextureFramePool(imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10);
            screen.Initialize(imageSource);

            //SetupAnnotationController(_imageClassifierResultAnnotationController, imageSource);

            var transformationOptions = imageSource.GetTransformationOptions();
            var flipHorizontally = transformationOptions.flipHorizontally;
            var flipVertically = transformationOptions.flipVertically;
            var imageProcessingOptions = new Tasks.Vision.Core.ImageProcessingOptions(rotationDegrees: (int)transformationOptions.rotationAngle);

            AsyncGPUReadbackRequest req = default;
            var waitUntilReqDone = new WaitUntil(() => req.done);
            var result = ImageClassifierResult.Alloc(5);

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
                    Debug.LogError($"Failed to read texture from the image source, exiting...");
                    break;
                }

                var image = textureFrame.BuildCPUImage();
                switch (taskApi.runningMode)
                {
                    case Tasks.Vision.Core.RunningMode.IMAGE:
                        if (taskApi.TryClassify(image, imageProcessingOptions, ref result))
                        {
                            //_imageClassifierResultAnnotationController.DrawNow(result);
                            Debug.Log("分类完成");
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
                            Debug.Log("分类完成");
                        }
                        else
                        {
                            //_imageClassifierResultAnnotationController.DrawNow(default);
                            Debug.Log("分类完成");
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
            Debug.Log("进入回调函数");
            //_imageClassifierResultAnnotationController.DrawLater(result);
        }
    }
}
