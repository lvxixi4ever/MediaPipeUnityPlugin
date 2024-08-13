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
        // ��ȡͼ���ʽ����ȡ��߶Ⱥ�ͨ����
        var format = inputImage.ImageFormat();
        int width = inputImage.Width();
        int height = inputImage.Height();
        int channels = inputImage.Channels();
        int widthStep = inputImage.Step();

        // ʹ�� PixelWriteLock ��ȡ��������
        using (var pixelWriteLock = new PixelWriteLock(inputImage))
        {
            IntPtr pixelDataPtr = pixelWriteLock.Pixels();
            int pixelDataSize = width * height * channels;

            // �ж�ͼ���ʽ�Ƿ�Ϊ�����ʽ
            bool isFloatFormat = format == ImageFormat.Types.Format.Srgba; // ����ʵ���������

            if (isFloatFormat)
            {
                // �� IntPtr ת��Ϊ float ����
                float* pixelData = (float*)pixelDataPtr.ToPointer();

                // ��һ����������
                for (int i = 0; i < pixelDataSize; i++)
                {
                    pixelData[i] /= 255.0f; // ��һ���� [0, 1] ��Χ
                }

                // ����һ���µ� NativeArray<byte> ���洢��һ���������
                NativeArray<byte> normalizedNativeArray = new NativeArray<byte>(pixelDataSize * sizeof(float), Allocator.Temp);

                // ����һ��������ݸ��Ƶ� NativeArray<byte>
                Buffer.MemoryCopy(pixelData, normalizedNativeArray.GetUnsafePtr(), pixelDataSize * sizeof(float), pixelDataSize * sizeof(float));

                // ʹ�ù�һ������������ݴ����µ� Mediapipe.Image ����
                var normalizedImage = new Mediapipe.Image(format, width, height, widthStep, normalizedNativeArray);
                normalizedNativeArray.Dispose();

                return normalizedImage;
            }
            else
            {
                // �Ǹ����ʽ�Ĵ���ʽ
                byte* pixelData = (byte*)pixelDataPtr.ToPointer();

                // ��һ������
                for (int i = 0; i < pixelDataSize; i++)
                {
                    pixelData[i] = (byte)(pixelData[i] / 255.0f * 255.0f); // ��һ���� [0, 1] ��Χ
                }

                // ����һ���µ� NativeArray<byte> ���洢��һ���������
                NativeArray<byte> normalizedNativeArray = new NativeArray<byte>(pixelDataSize, Allocator.Temp);

                // ����һ��������ݸ��Ƶ� NativeArray<byte>
                UnsafeUtility.MemCpy(normalizedNativeArray.GetUnsafePtr(), pixelData, pixelDataSize);

                // ʹ�ù�һ������������ݴ����µ� Mediapipe.Image ����
                var normalizedImage = new Mediapipe.Image(format, width, height, widthStep, normalizedNativeArray);
                normalizedNativeArray.Dispose();

                return normalizedImage;
            }
        }
    }
}
