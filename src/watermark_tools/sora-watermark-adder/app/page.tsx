'use client';

import { useState, useRef } from 'react';

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string>('');
  const [isConfirmed, setIsConfirmed] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState<string>('');
  const [message, setMessage] = useState('');
  
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 验证视频格式
    const allowedFormats = ['video/mp4', 'video/webm', 'video/mov', 'video/avi'];
    if (!allowedFormats.includes(file.type) && !file.name.match(/\.(mp4|webm|mov|avi)$/i)) {
      setMessage('不支持的视频格式！请上传 MP4, WebM, MOV 或 AVI 格式的视频。');
      return;
    }

    // 验证文件大小（50MB = 50 * 1024 * 1024 bytes）
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setMessage(`文件过大！文件大小为 ${(file.size / 1024 / 1024).toFixed(2)}MB，最大支持 50MB。`);
      return;
    }

    // 检查视频时长
    try {
      const duration = await getVideoDuration(file);
      if (duration > 60) {
        setMessage(`视频时长过长！视频时长为 ${duration.toFixed(1)} 秒，最长支持 60 秒（1分钟）。`);
        return;
      }

      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setIsConfirmed(false);
      setDownloadUrl('');
      setMessage(`视频已加载（${(file.size / 1024 / 1024).toFixed(2)}MB，${duration.toFixed(1)}秒），请预览并确认`);
    } catch (error) {
      setMessage('无法读取视频信息，请确保文件是有效的视频格式');
    }
  };

  const handleConfirm = () => {
    setIsConfirmed(true);
    setMessage('视频已确认，点击"添加水印"开始处理');
  };

  const handleCancel = () => {
    setVideoFile(null);
    setVideoPreview('');
    setIsConfirmed(false);
    setDownloadUrl('');
    setMessage('');
  };

  const getVideoDuration = async (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      
      video.onloadedmetadata = () => {
        URL.revokeObjectURL(video.src);
        resolve(video.duration);
      };

      video.onerror = () => {
        URL.revokeObjectURL(video.src);
        reject(new Error('无法加载视频'));
      };
      
      video.src = URL.createObjectURL(file);
    });
  };

  const detectVideoOrientation = async (file: File): Promise<'landscape' | 'portrait'> => {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      
      video.onloadedmetadata = () => {
        URL.revokeObjectURL(video.src);
        const width = video.videoWidth;
        const height = video.videoHeight;
        
        if (width > height) {
          resolve('landscape');
        } else {
          resolve('portrait');
        }
      };
      
      video.src = URL.createObjectURL(file);
    });
  };

  const processVideo = async () => {
    if (!videoFile || !isConfirmed) return;

    try {
      setProcessing(true);
      setProgress(10);
      setMessage('正在检测视频方向...');

      const orientation = await detectVideoOrientation(videoFile);
      
      setProgress(20);
      setMessage(`检测到${orientation === 'landscape' ? '横屏' : '竖屏'}视频，正在上传到服务器...`);

      // 创建表单数据
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('orientation', orientation);

      // 发送到服务端API
      const response = await fetch('/api/add-watermark', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '处理失败');
      }

      setProgress(80);
      setMessage('正在生成下载链接...');

      // 获取处理后的视频
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      setDownloadUrl(url);
      setMessage('视频处理完成！');
      setProcessing(false);
      setProgress(100);

    } catch (error) {
      console.error('处理视频时出错:', error);
      setMessage(`错误: ${error instanceof Error ? error.message : '处理失败'}`);
      setProcessing(false);
      setProgress(0);
    }
  };

  const handleDownload = () => {
    if (!downloadUrl) return;
    
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = `watermarked_${videoFile?.name || 'video.mp4'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h1 className="text-4xl font-bold text-center text-gray-800 mb-2">
            SORA视频水印工具
          </h1>
          <p className="text-center text-gray-600 mb-8">
            上传视频，自动添加适配的水印
          </p>

          {/* 上传区域 */}
          <div className="mb-8">
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-16 h-16 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mb-2 text-sm text-gray-500">
                  <span className="font-semibold">点击上传</span> 或拖拽视频文件
                </p>
                <p className="text-xs text-gray-400">支持 MP4, WebM, MOV, AVI 格式</p>
              </div>
              <input
                type="file"
                className="hidden"
                accept="video/mp4,video/webm,video/mov,video/avi,.mp4,.webm,.mov,.avi"
                onChange={handleFileChange}
                disabled={processing}
              />
            </label>
          </div>

          {/* 消息提示 */}
          {message && (
            <div className={`mb-6 p-4 rounded-lg ${
              message.includes('错误') ? 'bg-red-50 text-red-700' : 'bg-blue-50 text-blue-700'
            }`}>
              {message}
            </div>
          )}

          {/* 视频预览 */}
          {videoPreview && (
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-3 text-gray-800">视频预览</h2>
              <video
                ref={videoRef}
                src={videoPreview}
                controls
                className="w-full max-h-96 rounded-lg shadow-md"
              />
              
              {!isConfirmed && (
                <div className="flex gap-4 mt-4">
                  <button
                    onClick={handleConfirm}
                    className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    ✓ 确认视频
                  </button>
                  <button
                    onClick={handleCancel}
                    className="flex-1 bg-gray-300 hover:bg-gray-400 text-gray-700 font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    ✗ 取消
                  </button>
                </div>
              )}
            </div>
          )}

          {/* 处理按钮 */}
          {isConfirmed && !downloadUrl && (
            <div className="mb-6">
              <button
                onClick={processVideo}
                disabled={processing}
                className={`w-full py-4 px-6 rounded-lg text-white font-semibold text-lg transition-all ${
                  processing
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg'
                }`}
              >
                {processing ? '处理中...' : '🎬 添加水印'}
              </button>
            </div>
          )}

          {/* 进度条 */}
          {processing && (
            <div className="mb-6">
              <div className="bg-gray-200 rounded-full h-4 overflow-hidden">
                <div
                  className="bg-indigo-600 h-full transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-center text-sm text-gray-600 mt-2">{progress}%</p>
            </div>
          )}

          {/* 下载按钮 */}
          {downloadUrl && (
            <div className="space-y-4">
              <div className="bg-green-50 border-2 border-green-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-green-800 mb-2">✅ 处理完成！</h2>
                <p className="text-green-700 mb-4">您的视频已成功添加水印</p>
                <button
                  onClick={handleDownload}
                  className="w-full bg-green-500 hover:bg-green-600 text-white font-semibold py-4 px-6 rounded-lg transition-colors text-lg"
                >
                  ⬇️ 下载视频
                </button>
              </div>
              
              <button
                onClick={handleCancel}
                className="w-full bg-gray-300 hover:bg-gray-400 text-gray-700 font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                处理新视频
              </button>
            </div>
          )}

          {/* 使用说明 */}
          <div className="mt-8 p-6 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-3">📖 使用说明</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li>• 上传视频文件（支持 MP4, WebM, MOV, AVI 格式）</li>
              <li>• <span className="font-semibold text-orange-600">文件大小限制：最大 50MB</span></li>
              <li>• <span className="font-semibold text-orange-600">时长限制：最长 60 秒（1分钟）</span></li>
              <li>• 预览视频并确认</li>
              <li>• 系统会自动检测视频是横屏还是竖屏</li>
              <li>• 视频将在服务器端处理，自动去除水印中的黑色部分</li>
              <li>• 如果视频时长短于水印，水印会被截断</li>
              <li>• 如果视频时长长于水印，水印会循环播放</li>
              <li>• 处理完成后可以下载添加了水印的视频</li>
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}
