#!/bin/bash
# 停止并移除定时任务
for f in com.webullquant.tactical.plist com.webullquant.dashboard.plist; do
  launchctl unload ~/Library/LaunchAgents/$f 2>/dev/null || true
  rm -f ~/Library/LaunchAgents/$f
  echo "已移除 $f"
done
