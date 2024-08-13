using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe.Tasks.Components.Containers;
using ImageClassifierResult = Mediapipe.Tasks.Components.Containers.ClassificationResult;
using System;
namespace Mediapipe.Tasks.Vision.ImageClassifier
{
    public sealed class ImageClassifier : Core.BaseVisionTaskApi
    {
        private const string _IMAGE_IN_STREAM_NAME = "image_in";
        private const string _IMAGE_OUT_STREAM_NAME = "classifications";
        private const string _IMAGE_TAG = "IMAGE";
        private const string _NORM_RECT_STREAM_NAME = "norm_rect_in";
        private const string _CLASSIFICATIONS_STREAM_NAME = "classifications";
        private const string _CLASSIFICATIONS_TAG = "CLASSIFICATIONS";
        private const string _TASK_GRAPH_NAME = "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph";
        //::mediapipe::tasks::vision::image_classifier::ImageClassifierGraph
        private const string _TIMESTAMPED_CLASSIFICATIONS_STREAM_NAME = "timestamped_classifications_out";
        //private const string _TIMESTAMPED_CLASSIFICATIONS_TAG = "TIMESTAMPED_CLASSIFICATIONS";
       
        private const string _NORM_RECT_NAME = "norm_rect_in";
        private const string _NORM_RECT_TAG = "NORM_RECT";

        private const int _MICRO_SECONDS_PER_MILLISECOND = 1000;

        private readonly Tasks.Core.TaskRunner.PacketsCallback _packetCallback;
        private readonly NormalizedRect _normalizedRect = new NormalizedRect();

        private ImageClassifier(
          CalculatorGraphConfig graphConfig,
          Core.RunningMode runningMode,
          Tasks.Core.TaskRunner.PacketsCallback packetCallback) : base(graphConfig, runningMode, packetCallback)
        {
            _packetCallback = packetCallback;
        }

        public static ImageClassifier CreateFromModelPath(string modelPath)
        {
            var baseOptions = new Tasks.Core.BaseOptions(modelAssetPath: modelPath);
            var options = new ImageClassifierOptions(baseOptions, runningMode: Core.RunningMode.IMAGE);
            return CreateFromOptions(options);
        }

        public static ImageClassifier CreateFromOptions(ImageClassifierOptions options)
        {
           Debug.Log($"Creating ImageClassifier with options: {options}");

           var taskInfo = new Tasks.Core.TaskInfo<ImageClassifierOptions>(
                 taskGraph: _TASK_GRAPH_NAME,
            inputStreams: new List<string> {
               string.Join(":", _IMAGE_TAG, _IMAGE_IN_STREAM_NAME),
               //string.Join(":", _NORM_RECT_TAG, _NORM_RECT_STREAM_NAME),

                 },
            outputStreams: new List<string> {
                string.Join(":", _CLASSIFICATIONS_TAG, _CLASSIFICATIONS_STREAM_NAME),
                 //string.Join(":", _TIMESTAMPED_CLASSIFICATIONS, _TIMESTAMPED_CLASSIFICATIONS_STREAM_NAME),
                 },
             taskOptions: options);

             return new ImageClassifier(
                  taskInfo.GenerateGraphConfig(options.runningMode == Core.RunningMode.LIVE_STREAM),
                  options.runningMode,
                  BuildPacketsCallback(options));
       
        
        }

        public ImageClassifierResult Classify(Image image, Core.ImageProcessingOptions? imageProcessingOptions = null)
        {
            using var outputPackets = ClassifyInternal(image, imageProcessingOptions);

            var result = default(ImageClassifierResult);
            _ = TryBuildImageClassifierResult(outputPackets, ref result);
            return result;
        }

        public bool TryClassify(Image image, Core.ImageProcessingOptions? imageProcessingOptions, ref ImageClassifierResult result)
        {
            using var outputPackets = ClassifyInternal(image, imageProcessingOptions);
            return TryBuildImageClassifierResult(outputPackets, ref result);
        }

        private PacketMap ClassifyInternal(Image image, Core.ImageProcessingOptions? imageProcessingOptions)
        {
            Debug.Log($"Image shape: {image.Width()} x {image.Height()} x {image.Channels()}");
            ConfigureNormalizedRect(_normalizedRect, imageProcessingOptions, image, roiAllowed: false);

            var packetMap = new PacketMap();
            packetMap.Emplace(_IMAGE_IN_STREAM_NAME, Packet.CreateImage(image));
            

            return ProcessImageData(packetMap);
        }

        public ImageClassifierResult ClassifyForVideo(Image image, long timestampMillisec, Core.ImageProcessingOptions? imageProcessingOptions = null)
        {
            using var outputPackets = ClassifyForVideoInternal(image, timestampMillisec, imageProcessingOptions);

            var result = default(ImageClassifierResult);
            _ = TryBuildImageClassifierResult(outputPackets, ref result);
            return result;
        }

        public bool TryClassifyForVideo(Image image, long timestampMillisec, Core.ImageProcessingOptions? imageProcessingOptions, ref ImageClassifierResult result)
        {
            using var outputPackets = ClassifyForVideoInternal(image, timestampMillisec, imageProcessingOptions);
            return TryBuildImageClassifierResult(outputPackets, ref result);
        }

        private PacketMap ClassifyForVideoInternal(Image image, long timestampMillisec, Core.ImageProcessingOptions? imageProcessingOptions = null)
        {
            ConfigureNormalizedRect(_normalizedRect, imageProcessingOptions, image, roiAllowed: false);
            var timestampMicrosec = timestampMillisec * _MICRO_SECONDS_PER_MILLISECOND;

            // ¼ì²éÍ¼ÏñÐÎ×´
            //Debug.Log($"Image shape: {image.Width()} x {image.Height()} x {image.Channels()}");

            var packetMap = new PacketMap();
            packetMap.Emplace(_IMAGE_IN_STREAM_NAME, Packet.CreateImageAt(image, timestampMicrosec));
            

            return ProcessVideoData(packetMap);
        }

        public void ClassifyAsync(Image image, long timestampMillisec, Core.ImageProcessingOptions? imageProcessingOptions = null)
        {
            ConfigureNormalizedRect(_normalizedRect, imageProcessingOptions, image, roiAllowed: false);
            var timestampMicrosec = timestampMillisec * _MICRO_SECONDS_PER_MILLISECOND;

            var packetMap = new PacketMap();
            packetMap.Emplace(_IMAGE_IN_STREAM_NAME, Packet.CreateImageAt(image, timestampMicrosec));
            

            SendLiveStreamData(packetMap);
        }

        //private bool TryBuildImageClassifierResult(PacketMap outputPackets, ref ImageClassifierResult result)
        //    => TryBuildImageClassifierResult(outputPackets, ref result);

        private static bool TryBuildImageClassifierResult(PacketMap outputPackets, ref ImageClassifierResult result)
        {
            using var ClassiferPacket = outputPackets.At<ImageClassifierResult>(_IMAGE_OUT_STREAM_NAME);
            if (ClassiferPacket.IsEmpty())
            {
                return false;
            }
            ClassiferPacket.Get(ref result);
            return true;
        }

        private static Tasks.Core.TaskRunner.PacketsCallback BuildPacketsCallback(ImageClassifierOptions options)
        {
            var resultCallback = options.resultCallback;
            if (resultCallback == null)
            {
                return null;
            }

            var result = ImageClassifierResult.Alloc(options.maxResults ?? 0);

            return (PacketMap outputPackets) =>
            {
                using var outPacket = outputPackets.At<ImageClassifierResult>(_CLASSIFICATIONS_STREAM_NAME);
                if (outPacket == null || outPacket.IsEmpty())
                {
                    return;
                }

                outPacket.Get(ref result);
                var timestamp = outPacket.TimestampMicroseconds() / _MICRO_SECONDS_PER_MILLISECOND;

                resultCallback(result,timestamp);
            };
        }

        //private static Tasks.Core.TaskRunner.PacketsCallback BuildPacketsCallback(ImageClassifierOptions options)
        //{
        //    if (options.runningMode == Core.RunningMode.LIVE_STREAM && options.resultCallback != null)
        //    {
        //        return new Tasks.Core.TaskRunner.PacketsCallback((PacketMap outputPackets) =>
        //        {
        //            var classificationResult = new ImageClassifierResult();
        //            if (!TryBuildImageClassifierResult(outputPackets, ref classificationResult))
        //            {
        //                return;
        //            }
        //            var image = outputPackets.At<Image>(_IMAGE_OUT_STREAM_NAME);
        //            var timestamp = outputPackets.At<Image>(_IMAGE_OUT_STREAM_NAME).Timestamp().Microseconds() / _MICRO_SECONDS_PER_MILLISECOND;
        //            options.resultCallback(classificationResult, image, timestamp);
        //        });
        //    }
        //    return null;
        //}
    }
}
