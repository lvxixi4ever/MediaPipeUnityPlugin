using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Mediapipe.Tasks.Core;
using Mediapipe.Tasks.Components.Containers;
using Mediapipe.Tasks.Vision;

using ImageClassifierResult = Mediapipe.Tasks.Components.Containers.ClassificationResult;

namespace Mediapipe.Tasks.Vision.ImageClassifier
{
    /// <summary>
    /// Options for the image classifier task.
    /// </summary>
    /// 
    ///for image classity
    public class NormalizationOptions
    {
        public float[] Mean { get; set; }
        public float[] Std { get; set; }

        public NormalizationOptions(float[] mean, float[] std)
        {
            Mean = mean;
            Std = std;
        }
    }
    public sealed class ImageClassifierOptions : Tasks.Core.ITaskOptions
    {
        /// <summary>
        /// The delegate for handling classification results.
        /// </summary>
        /// <param name="classificationResult">The classification results.</param>
        /// <param name="image">The input image that the classifier runs on.</param>
        /// <param name="timestampMillisec">The input timestamp in milliseconds.</param>
        //public delegate void ResultCallback(ImageClassifierResult classificationResult, Image image, long timestampMillisec);
        public delegate void ResultCallback(ImageClassifierResult classificationResult,long timestampMillisec);

        ///image classify
        //public NormalizationOptions NormalizeOptions { get; }

        /// <summary>
        /// Base options for the image classifier task.
        /// </summary>
        public BaseOptions baseOptions { get; }

        /// <summary>
        /// Options for configuring the classifier behavior, such as score threshold,
        /// number of results, etc.
        /// </summary>
        //public ClassifierOptions classifierOptions { get; }

        /// <summary>
        ///   The maximum number of top-scored classification results to return.
        /// </summary>
        public int? maxResults { get; }

        /// <summary>
        ///   Overrides the ones provided in the model metadata. Results below this value are rejected.
        /// </summary>
        public float? scoreThreshold { get; }

        /// <summary>
        /// The running mode of the task. Default to the image mode.
        /// ImageClassifier has three running modes:
        /// <list type="number">
        ///   <item>
        ///     <description>The image mode for classifying on single image inputs.</description>
        ///   </item>
        ///   <item>
        ///     <description>The video mode for classifying on the decoded frames of a video.</description>
        ///   </item>
        ///   <item>
        ///     <description>
        ///       The live stream mode for classifying on the live stream of input data, such as from a camera.
        ///       In this mode, the <see cref="resultCallback" /> must be specified to receive the classification results asynchronously.
        ///     </description>
        ///   </item>
        /// </list>
        /// </summary>
        public Core.RunningMode runningMode { get; }
        /// <summary>
        /// The maximum number of classification results.
        /// </summary>
        //public int maxNumResults { get; }

        /// <summary>
        /// The minimum confidence score for a classification to be considered successful.
        /// </summary>
        //public float minClassificationConfidence { get; }
        /// <summary>
        /// The user-defined result callback for processing live stream data.
        /// The result callback should only be specified when the running mode is set to the live stream mode.
        /// </summary>
        public ResultCallback resultCallback { get; }

        public ImageClassifierOptions(
            BaseOptions baseOptions,
            Core.RunningMode runningMode = Core.RunningMode.IMAGE,
            int? maxResults = null,
            float? scoreThreshold = null,
            //NormalizationOptions normalizeOptions = null,
            ResultCallback resultCallback = null)
        {
            this.baseOptions = baseOptions;
            this.runningMode = runningMode;
            this.maxResults = maxResults;
            this.scoreThreshold = scoreThreshold;
            //this.NormalizeOptions = normalizeOptions;
            this.resultCallback = resultCallback;
        }

        internal Proto.ImageClassifierGraphOptions ToProto()
        {
            var baseOptionsProto = baseOptions.ToProto();
            baseOptionsProto.UseStreamMode = runningMode != Core.RunningMode.IMAGE;

            var classifierOptions = new Components.Processors.Proto.ClassifierOptions();

            if (maxResults is int maxResultsValue)
            {
                classifierOptions.MaxResults = maxResultsValue;
            }
            if (scoreThreshold is float scoreThresholdValue)
            {
                classifierOptions.ScoreThreshold = scoreThresholdValue;
            }

            //var normalizationOptions = new NormalizationOptions();
            //if (NormalizeOptions != null)
            //{
            //    normalizationOptions.Mean.Add(NormalizeOptions.Mean);
            //    normalizationOptions.Std.Add(NormalizeOptions.Std);
            //}

            return new Proto.ImageClassifierGraphOptions
            {
                BaseOptions = baseOptionsProto,
                ClassifierOptions = classifierOptions
            };
        }

        CalculatorOptions ITaskOptions.ToCalculatorOptions()
        {
            var options = new CalculatorOptions();
            options.SetExtension(Proto.ImageClassifierGraphOptions.Extensions.Ext, ToProto());
            return options;
        }
    }
}
