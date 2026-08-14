# 安装指南

## 快速开始

### 1. 安装 FFmpeg（必需）

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Windows
1. 访问 https://ffmpeg.org/download.html
2. 下载适合Windows的版本
3. 解压到一个目录，例如 `C:\ffmpeg`
4. 将 `C:\ffmpeg\bin` 添加到系统环境变量 PATH

### 2. 验证 FFmpeg 安装
```bash
ffmpeg -version
```

应该看到FFmpeg的版本信息。

### 3. 安装项目依赖
```bash
cd video-watermark-app
npm install
```

### 4. 运行项目
```bash
npm run dev
```

### 5. 访问应用
打开浏览器访问: http://localhost:3000

## 常见问题

**Q: 提示找不到 FFmpeg？**
A: 确保FFmpeg已安装并在系统PATH中。运行 `which ffmpeg` (macOS/Linux) 或 `where ffmpeg` (Windows) 检查。

**Q: 视频处理失败？**
A: 检查服务器日志，确认FFmpeg版本是否支持所需的编解码器。

**Q: 能处理多大的视频？**
A: 取决于服务器配置，建议限制在500MB以内以获得最佳性能。

