#!/bin/bash
# 安装 / 更新定时任务：战术层（工作日 08:45 启动）+ 仪表盘（常驻）
set -e
cd "$(dirname "$0")"
mkdir -p ~/Library/LaunchAgents ../logs
pkill -f "wquant dashboard" 2>/dev/null || true   # 手动开的仪表盘先关掉，避免端口冲突
for f in com.webullquant.tactical.plist com.webullquant.dashboard.plist; do
  launchctl unload ~/Library/LaunchAgents/$f 2>/dev/null || true
  cp $f ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/$f
  echo "已安装 $f"
done
launchctl list | grep webullquant
