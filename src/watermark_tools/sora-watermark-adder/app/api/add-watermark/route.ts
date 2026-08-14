import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// 确保临时目录存在
const TEMP_DIR = path.join(process.cwd(), 'temp');

async function ensureTempDir() {
  if (!existsSync(TEMP_DIR)) {
    await mkdir(TEMP_DIR, { recursive: true });
  }
}

// 获取视频时长
function getVideoDuration(filePath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) {
        reject(err);
      } else {
        resolve(metadata.format.duration || 0);
      }
    });
  });
}

// 获取视频尺寸
function getVideoSize(filePath: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) {
        reject(err);
      } else {
        const videoStream = metadata.streams.find(s => s.codec_type === 'video');
        if (videoStream && videoStream.width && videoStream.height) {
          resolve({ width: videoStream.width, height: videoStream.height });
        } else {
          reject(new Error('无法获取视频尺寸'));
        }
      }
    });
  });
}

// 处理视频添加水印 - 使用colorkey去除黑色，直接调用ffmpeg
async function processVideoWithWatermark(
  inputPath: string,
  watermarkPath: string,
  outputPath: string,
  videoDuration: number,
  watermarkDuration: number,
  videoSize: { width: number; height: number }
): Promise<void> {
  // 计算需要循环的次数
  const loopCount = Math.ceil(videoDuration / watermarkDuration);
  
  console.log(`需要循环水印 ${loopCount} 次`);
  
  // 构建filter_complex字符串
  // 使用colorkey去除黑色，然后overlay叠加
  let filterComplex: string;
  
  if (loopCount === 1) {
    // 水印比视频长，直接截断水印
    filterComplex = `[1:v]scale=${videoSize.width}:${videoSize.height}[scaled];[scaled]colorkey=0x000000:0.3:0.2[keyed];[keyed]trim=end=${videoDuration},setpts=PTS-STARTPTS[wm];[0:v][wm]overlay=0:0[v]`;
  } else {
    // 视频比水印长，需要循环水印
    filterComplex = `[1:v]scale=${videoSize.width}:${videoSize.height}[scaled];[scaled]colorkey=0x000000:0.3:0.2[keyed];[0:v][keyed]overlay=0:0[v]`;
  }
  
  // 构建完整的ffmpeg命令
  // 关键：-stream_loop必须在对应的-i之前
  const args = [
    'ffmpeg',
    '-i', `"${inputPath}"`,  // 主视频输入
  ];
  
  // 如果需要循环，在水印输入之前添加-stream_loop
  if (loopCount > 1) {
    args.push('-stream_loop', String(loopCount - 1));
  }
  
  args.push(
    '-i', `"${watermarkPath}"`,  // 水印输入
    '-filter_complex', `"${filterComplex}"`,
    '-map', '[v]',
    '-map', '0:a?',
    '-c:a', 'copy',
    '-c:v', 'libx264',
    '-preset', 'medium',
    '-crf', '23',
    '-t', String(videoDuration),
    '-y',  // 覆盖输出文件
    `"${outputPath}"`
  );
  
  const command = args.join(' ');
  console.log('执行FFmpeg命令:', command);
  
  try {
    const { stdout, stderr } = await execAsync(command, {
      maxBuffer: 50 * 1024 * 1024  // 50MB buffer
    });
    
    if (stderr) {
      console.log('FFmpeg stderr:', stderr);
    }
    console.log('视频处理完成');
  } catch (error) {
    console.error('FFmpeg执行错误:', error);
    throw error;
  }
}

export async function POST(request: NextRequest) {
  const tempFiles: string[] = [];

  try {
    // 确保临时目录存在
    await ensureTempDir();

    // 解析表单数据
    const formData = await request.formData();
    const videoFile = formData.get('video') as File;
    const orientation = formData.get('orientation') as string;

    if (!videoFile) {
      return NextResponse.json({ error: '未上传视频文件' }, { status: 400 });
    }

    // 验证文件大小（50MB）
    const maxSize = 50 * 1024 * 1024;
    if (videoFile.size > maxSize) {
      return NextResponse.json({ 
        error: `文件过大！文件大小为 ${(videoFile.size / 1024 / 1024).toFixed(2)}MB，最大支持 50MB。` 
      }, { status: 400 });
    }

    if (!orientation || !['landscape', 'portrait'].includes(orientation)) {
      return NextResponse.json({ error: '无效的视频方向' }, { status: 400 });
    }

    // 生成临时文件路径
    const timestamp = Date.now();
    const inputPath = path.join(TEMP_DIR, `input_${timestamp}.mp4`);
    const outputPath = path.join(TEMP_DIR, `output_${timestamp}.mp4`);
    tempFiles.push(inputPath, outputPath);

    // 保存上传的视频
    const videoBuffer = Buffer.from(await videoFile.arrayBuffer());
    await writeFile(inputPath, videoBuffer);

    // 选择水印文件
    const watermarkFileName = orientation === 'landscape' ? 'water_横屏.mp4' : 'water_竖屏.mp4';
    const watermarkPath = path.join(process.cwd(), 'public', 'watermarks', watermarkFileName);

    if (!existsSync(watermarkPath)) {
      return NextResponse.json({ error: '水印文件不存在' }, { status: 500 });
    }

    // 获取视频时长和尺寸
    const videoDuration = await getVideoDuration(inputPath);
    const watermarkDuration = await getVideoDuration(watermarkPath);
    const videoSize = await getVideoSize(inputPath);

    // 验证视频时长（60秒）
    if (videoDuration > 60) {
      // 清理已上传的文件
      await unlink(inputPath);
      return NextResponse.json({ 
        error: `视频时长过长！视频时长为 ${videoDuration.toFixed(1)} 秒，最长支持 60 秒（1分钟）。` 
      }, { status: 400 });
    }

    console.log(`视频时长: ${videoDuration}秒, 水印时长: ${watermarkDuration}秒`);
    console.log(`视频尺寸: ${videoSize.width}x${videoSize.height}`);

    // 处理视频
    await processVideoWithWatermark(
      inputPath,
      watermarkPath,
      outputPath,
      videoDuration,
      watermarkDuration,
      videoSize
    );

    // 读取处理后的视频
    const { readFile } = await import('fs/promises');
    const outputBuffer = await readFile(outputPath);
    
    // 清理临时文件
    for (const file of tempFiles) {
      try {
        await unlink(file);
      } catch (err) {
        console.error(`清理文件失败: ${file}`, err);
      }
    }

    // 生成安全的文件名（移除非ASCII字符）
    const safeFilename = `watermarked_${Date.now()}.mp4`;

    // 返回处理后的视频
    // 将 Buffer 转换为 Uint8Array，这是 Response 支持的类型
    const uint8Array = new Uint8Array(outputBuffer);
    
    const headers = new Headers();
    headers.set('Content-Type', 'video/mp4');
    headers.set('Content-Disposition', `attachment; filename="${safeFilename}"`);
    headers.set('Content-Length', outputBuffer.length.toString());
    
    return new Response(uint8Array, {
      status: 200,
      headers: headers,
    });
  } catch (error) {
    // 清理临时文件
    for (const file of tempFiles) {
      try {
        await unlink(file);
      } catch (err) {
        // 忽略清理错误
      }
    }

    console.error('处理视频时出错:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : '处理视频失败' },
      { status: 500 }
    );
  }
}

