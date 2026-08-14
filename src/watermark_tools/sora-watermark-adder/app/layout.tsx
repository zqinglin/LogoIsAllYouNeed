import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "sora 水印添加 - 自动添加sora水印",
  description: "上传视频，自动检测横竖屏并添加适配的水印",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
