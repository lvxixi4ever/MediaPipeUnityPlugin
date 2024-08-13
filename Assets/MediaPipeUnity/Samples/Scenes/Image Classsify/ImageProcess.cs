using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Mediapipe;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

public class ImageProcessor
{
    public static unsafe Mediapipe.Image NormalizeImage(Mediapipe.Image inputImage)
    {
        // 获取图像格式、宽度、高度和通道数
        var format = inputImage.ImageFormat();
        int width = inputImage.Width();
        int height = inputImage.Height();
        int channels = inputImage.Channels();
        int widthStep = inputImage.Step();

        // 使用 PixelWriteLock 获取像素数据
        using (var pixelWriteLock = new PixelWriteLock(inputImage))
        {
            IntPtr pixelDataPtr = pixelWriteLock.Pixels();
            int pixelDataSize = width * height * channels;

            // 判断图像格式是否为浮点格式
            bool isFloatFormat = format == ImageFormat.Types.Format.Srgba; // 根据实际情况调整

            if (isFloatFormat)
            {
                // 将 IntPtr 转换为 float 数组
                float* pixelData = (float*)pixelDataPtr.ToPointer();

                // 归一化浮点数据
                for (int i = 0; i < pixelDataSize; i++)
                {
                    pixelData[i] /= 255.0f; // 归一化到 [0, 1] 范围
                }

                // 创建一个新的 NativeArray<byte> 来存储归一化后的数据
                NativeArray<byte> normalizedNativeArray = new NativeArray<byte>(pixelDataSize * sizeof(float), Allocator.Temp);

                // 将归一化后的数据复制到 NativeArray<byte>
                Buffer.MemoryCopy(pixelData, normalizedNativeArray.GetUnsafePtr(), pixelDataSize * sizeof(float), pixelDataSize * sizeof(float));

                // 使用归一化后的像素数据创建新的 Mediapipe.Image 对象
                var normalizedImage = new Mediapipe.Image(format, width, height, widthStep, normalizedNativeArray);
                normalizedNativeArray.Dispose();

                return normalizedImage;
            }
            else
            {
                // 非浮点格式的处理方式
                byte* pixelData = (byte*)pixelDataPtr.ToPointer();

                // 归一化数据
                for (int i = 0; i < pixelDataSize; i++)
                {
                    pixelData[i] = (byte)(pixelData[i] / 255.0f * 255.0f); // 归一化到 [0, 1] 范围
                }

                // 创建一个新的 NativeArray<byte> 来存储归一化后的数据
                NativeArray<byte> normalizedNativeArray = new NativeArray<byte>(pixelDataSize, Allocator.Temp);

                // 将归一化后的数据复制到 NativeArray<byte>
                UnsafeUtility.MemCpy(normalizedNativeArray.GetUnsafePtr(), pixelData, pixelDataSize);

                // 使用归一化后的像素数据创建新的 Mediapipe.Image 对象
                var normalizedImage = new Mediapipe.Image(format, width, height, widthStep, normalizedNativeArray);
                normalizedNativeArray.Dispose();

                return normalizedImage;
            }
        }
    }
}
