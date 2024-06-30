// Copyright (c) 2021 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

using System.Collections;
using System.IO;
using UnityEngine;

namespace Mediapipe.Unity
{
  public class AssetBundleResourceManager : IResourceManager
  {
    private static readonly string _TAG = nameof(AssetBundleResourceManager);

    private static string _AssetBundlePath;
    private static string _CachePathRoot;

    public AssetBundleResourceManager(string assetBundleName, string cachePath = "Cache")
    {
      ResourceUtil.EnableCustomResolver();
      _AssetBundlePath = Path.Combine(Application.streamingAssetsPath, assetBundleName);
      _CachePathRoot = Path.Combine(Application.persistentDataPath, cachePath);
    }

    private AssetBundleCreateRequest _assetBundleReq;
    private AssetBundle assetBundle => _assetBundleReq?.assetBundle;

    public void ClearAllCacheFiles()
    {
      if (Directory.Exists(_CachePathRoot))
      {
        Directory.Delete(_CachePathRoot, true);
      }
    }

    public IEnumerator LoadAssetBundleAsync()
    {
      if (assetBundle != null)
      {
        Logger.LogWarning(_TAG, "AssetBundle is already loaded");
        yield break;
      }

      // No need to lock because this code can be run in main thread only.
      _assetBundleReq = AssetBundle.LoadFromFileAsync(_AssetBundlePath);
      yield return _assetBundleReq;

      if (_assetBundleReq.assetBundle == null)
      {
        throw new IOException($"Failed to load {_AssetBundlePath}");
      }
    }

    IEnumerator IResourceManager.PrepareAssetAsync(string name, string uniqueKey, bool overwrite)
    {
      var destFilePath = GetCachePathFor(uniqueKey);
      if (overwrite)
      {
        ResourceUtil.SetAssetPath(name, destFilePath);
      }
      else
      {
        ResourceUtil.AddAssetPath(name, destFilePath);
      }

      if (File.Exists(destFilePath) && !overwrite)
      {
        Logger.LogInfo(_TAG, $"{name} will not be copied to {destFilePath} because it already exists");
        yield break;
      }

      if (assetBundle == null)
      {
        yield return LoadAssetBundleAsync();
      }

      var assetLoadReq = assetBundle.LoadAssetAsync<TextAsset>(name);
      yield return assetLoadReq;

      if (assetLoadReq.asset == null)
      {
        throw new IOException($"Failed to load {name} from {assetBundle.name}");
      }

      Logger.LogVerbose(_TAG, $"Writing {name} data to {destFilePath}...");
      if (!Directory.Exists(_CachePathRoot))
      {
        var _ = Directory.CreateDirectory(_CachePathRoot);
      }
      var bytes = (assetLoadReq.asset as TextAsset).bytes;
      File.WriteAllBytes(destFilePath, bytes);
      Logger.LogVerbose(_TAG, $"{name} is saved to {destFilePath} (length={bytes.Length})");
    }

    private static string GetCachePathFor(string assetName)
    {
      return Path.Combine(_CachePathRoot, assetName);
    }
  }
}
