# sora水印添加工具

自动化为你的视频添加sora水印

在线demo：
https://sora.snowywar.top/ (不会定时维护，不保证随时可用)

## 前置要求

### 必须安装 FFmpeg

本项目需要在服务器上安装 FFmpeg。

#### macOS 安装:
```bash
brew install ffmpeg
```

#### Ubuntu/Debian 安装:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### CentOS/RHEL 安装:
```bash
sudo yum install epel-release
sudo yum install ffmpeg
```

#### Windows 安装:
1. 从 [FFmpeg官网](https://ffmpeg.org/download.html) 下载
2. 解压并添加到系统 PATH

#### 验证安装:
```bash
ffmpeg -version
```

## 安装和运行

1. 安装依赖:
```bash
npm install
```

2. 确保水印视频已放置在正确位置:
```
public/watermarks/
├── water_横屏.mp4
└── water_竖屏.mp4
```

3. 运行开发服务器:
```bash
npm run dev
```

4. 在浏览器中打开 [http://localhost:3000](http://localhost:3000)

## docker运行

```
docker pull snowywar/sora-watermark:latest
docker run -d -p 3000:3000 snowywar/sora-watermark:latest
```
或者
```
git clone https://github.com/jiayuqi7813/sora-watermark-adder.git
cd sora-watermark-adder
docker compose up -d
```


## 项目结构

```
video-watermark-app/
├── app/
│   ├── page.tsx              # 主应用页面（前端）
│   ├── layout.tsx            # 布局配置
│   ├── globals.css           # 全局样式
│   └── api/
│       └── add-watermark/
│           └── route.ts      # API路由（服务端视频处理）
├── public/
│   └── watermarks/
│       ├── water_横屏.mp4    # 横屏水印
│       └── water_竖屏.mp4    # 竖屏水印
├── temp/                     # 临时文件目录（自动创建）
├── next.config.ts            # Next.js配置
└── README.md                 # 本文件
```

