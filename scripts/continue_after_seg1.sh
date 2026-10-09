#!/bin/bash
# 等 seg1 完成 -> 程序化质检 -> seg2 -> 拼接
cd "$HOME/video10s" || exit 1
echo "[watch] 等待 seg1 写入 state.json ..."
ok=0
for i in $(seq 1 300); do
  if python3 -c "
import json,sys
try:
    s=json.load(open('state.json'))
    v=s.get('seg1',{}).get('video') or {}
    good = s.get('seg1',{}).get('status')=='success' and v.get('black_frames',1)==0 and not v.get('nan',True) and v.get('brightness_min',0)>20
    sys.exit(0 if good else 1)
except Exception:
    sys.exit(1)
"; then ok=1; break; fi
  sleep 10
done
if [ "$ok" != "1" ]; then echo "[watch] seg1 未通过质检或超时，停止"; exit 2; fi
echo "[watch] seg1 质检通过，开始 seg2"
python3 driver10s.py seg2 || { echo "[watch] seg2 失败"; exit 3; }
echo "[watch] seg2 完成，开始拼接"
python3 driver10s.py assemble || { echo "[watch] 拼接失败"; exit 4; }
echo "[watch] 全流程完成"