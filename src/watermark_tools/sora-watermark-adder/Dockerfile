FROM node:20-alpine AS base

# 安装 FFmpeg 和其他必要的依赖
RUN apk add --no-cache \
    ffmpeg \
    libc6-compat

WORKDIR /app

# 复制 package 文件
COPY package*.json ./

# 安装依赖
RUN npm ci

# 复制项目文件
COPY . .

# 构建 Next.js 应用
RUN npm run build

# 暴露端口
EXPOSE 3000

# 设置环境变量
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=::

# 启动应用 (监听IPv6，同时也支持IPv4)
CMD ["npm", "start"]

