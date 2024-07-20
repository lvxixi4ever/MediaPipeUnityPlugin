// Copyright (c) 2023 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.Collections;
using UnityEngine;

using Mediapipe.Tasks.Vision.FaceLandmarker;//在packages Tasks里面
using UnityEngine.Rendering;

namespace Mediapipe.Unity.Sample.FaceLandmarkDetection//Assets Tasks里面
{
  public class FaceLandmarkerRunner : VisionTaskApiRunner<FaceLandmarker>
  {
    [SerializeField] private FaceLandmarkerResultAnnotationController _faceLandmarkerResultAnnotationController;

    private Experimental.TextureFramePool _textureFramePool;

    public readonly FaceLandmarkDetectionConfig config = new FaceLandmarkDetectionConfig();

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
      Debug.Log($"NumFaces = {config.NumFaces}");
      Debug.Log($"MinFaceDetectionConfidence = {config.MinFaceDetectionConfidence}");
      Debug.Log($"MinFacePresenceConfidence = {config.MinFacePresenceConfidence}");
      Debug.Log($"MinTrackingConfidence = {config.MinTrackingConfidence}");
      Debug.Log($"OutputFaceBlendshapes = {config.OutputFaceBlendshapes}");
      Debug.Log($"OutputFacialTransformationMatrixes = {config.OutputFacialTransformationMatrixes}");

      yield return AssetLoader.PrepareAssetAsync(config.ModelPath);

      var options = config.GetFaceLandmarkerOptions(config.RunningMode == Tasks.Vision.Core.RunningMode.LIVE_STREAM ? OnFaceLandmarkDetectionOutput : null);//设置回调函数
      taskApi = FaceLandmarker.CreateFromOptions(options);
 
      var imageSource = ImageSourceProvider.ImageSource;//摄像头的话返回的是WebCamSource实例
      //管理并提供不同类型的图像源,这些图像源可以是网络摄像头（WebCamSource）、静态图片（StaticImageSource）或者视频文件（VideoSource）
      yield return imageSource.Play();

      if (!imageSource.isPrepared)
      {
        Debug.LogError("Failed to start ImageSource, exiting...");
        yield break;
      }

      // Use RGBA32 as the input format.
      // TODO: When using GpuBuffer, MediaPipe assumes that the input format is BGRA, so maybe the following code needs to be fixed.
      _textureFramePool = new Experimental.TextureFramePool(imageSource.textureWidth, imageSource.textureHeight, TextureFormat.RGBA32, 10);

      // NOTE: The screen will be resized later, keeping the aspect ratio.
      screen.Initialize(imageSource);

      SetupAnnotationController(_faceLandmarkerResultAnnotationController, imageSource);

      var transformationOptions = imageSource.GetTransformationOptions();
      var flipHorizontally = transformationOptions.flipHorizontally;
      var flipVertically = transformationOptions.flipVertically;
      var imageProcessingOptions = new Tasks.Vision.Core.ImageProcessingOptions(rotationDegrees: (int)transformationOptions.rotationAngle);

      AsyncGPUReadbackRequest req = default;
      var waitUntilReqDone = new WaitUntil(() => req.done);//() => req.done
      var result = FaceLandmarkerResult.Alloc(options.numFaces);

      while (true)
      {
        if (isPaused)
        {
          yield return new WaitWhile(() => isPaused);
        }

        if (!_textureFramePool.TryGetTextureFrame(out var textureFrame))//从池中获取一个空闲的TextureFrame，并将WebCamSource的当前帧复制到这个TextureFrame中
                {
          yield return new WaitForEndOfFrame();
          continue;
        }

        // Copy current image to TextureFrame，还有从GPU读取图片等操作
        req = textureFrame.ReadTextureAsync(imageSource.GetCurrentTexture(), flipHorizontally, flipVertically);//imageSource.GetCurrentTexture()返回webCamTexture类实例，获取当前帧
           yield return waitUntilReqDone;

        if (req.hasError)
        {
          Debug.LogError($"Failed to read texture from the image source, exiting...");
          break;
        }

        var image = textureFrame.BuildCPUImage();//构建CPUImage对象，包含了具体的图像数据
        switch (taskApi.runningMode)
        {
          case Tasks.Vision.Core.RunningMode.IMAGE:
            if (taskApi.TryDetect(image, imageProcessingOptions, ref result))
            {
              _faceLandmarkerResultAnnotationController.DrawNow(result);
            }
            else
            {
              _faceLandmarkerResultAnnotationController.DrawNow(default);
            }
            break;
          case Tasks.Vision.Core.RunningMode.VIDEO:
            if (taskApi.TryDetectForVideo(image, GetCurrentTimestampMillisec(), imageProcessingOptions, ref result))
            {
              _faceLandmarkerResultAnnotationController.DrawNow(result);
            }
            else
            {
              _faceLandmarkerResultAnnotationController.DrawNow(default);
            }
            break;
          case Tasks.Vision.Core.RunningMode.LIVE_STREAM:
            taskApi.DetectAsync(image, GetCurrentTimestampMillisec(), imageProcessingOptions);
            break;
        }

        textureFrame.Release();
      }
    }

    private void OnFaceLandmarkDetectionOutput(FaceLandmarkerResult result, Image image, long timestamp)
    {
      Debug.Log("进入回调函数");
      _faceLandmarkerResultAnnotationController.DrawLater(result);
    }
  }
}
