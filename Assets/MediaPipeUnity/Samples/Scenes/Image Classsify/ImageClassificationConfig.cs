using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Mediapipe.Tasks.Vision.ImageClassifier;
using Mediapipe.Tasks.Core;
using System;


namespace Mediapipe.Unity.Sample.ImageClassification
{
    //public class TaskApiConfig
    //{
    //    public string ModelPath { get; set; }
    //    public RunningMode RunningMode { get; set; } = Tasks.Vision.Core.RunningMode.IMAGE;
    //    public Delegate Delegate { get; set; } = Delegate.CPU;

    //    public BaseOptions GetBaseOptions()
    //    {
    //        return new BaseOptions
    //        {
    //            ModelAssetPath = ModelPath,
    //            Delegate = Delegate
    //        };
    //    }
    //}
    public class ImageClassificationConfig
    {
        public Tasks.Core.BaseOptions.Delegate Delegate { get; set; } =
#if     UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
         Tasks.Core.BaseOptions.Delegate.CPU;
#else
          Tasks.Core.BaseOptions.Delegate.GPU;
#endif
        public Tasks.Vision.Core.RunningMode RunningMode { get; set; } = Tasks.Vision.Core.RunningMode.IMAGE;
        //public Tasks.Vision.Core.RunningMode RunningMode { get; set; } = Tasks.Vision.Core.RunningMode.LIVE_STREAM;
        public int maxResults { get; set; } = 5;
        public float scoreThreshold { get; set; } = 0.3f;

        //public string ModelPath => "face_landmarker_v2_with_blendshapes.bytes" ;
        //public string ModelPath => "mobilenet.bytes";

        public string ModelPath => "emotion_big_yolov8_n_120_float32_with_metadata.bytes";
        //public string ModelPath => "emotion_big_yolov8_m_150_with_metadata.bytes";
        //public string ModelPath => "effi1.bytes";
        public ImageClassifierOptions GetImageClassifierOptions(ImageClassifierOptions.ResultCallback resultCallback = null)
        {
            return new ImageClassifierOptions(
              new Tasks.Core.BaseOptions(Delegate, modelAssetPath: ModelPath),
              runningMode: RunningMode,
              maxResults : maxResults,
              scoreThreshold: scoreThreshold,
              resultCallback: resultCallback
            );
      
        }



    }
}

