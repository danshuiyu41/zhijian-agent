#!/usr/bin/env bash
# =============================================================================
# 智检 Agent · 云服务器快速部署（不训练，直接装现成模型）
# -----------------------------------------------------------------------------
# 和 deploy.sh 的区别：
#   deploy.sh      从零开始，要在服务器上重新训练 40 分钟，吃 4GB 内存
#   deploy_fast.sh 直接用安装包里训练好的权重，3~5 分钟装完，1GB 内存就够
#
# 用法（在服务器上，已把 zhijian-agent-v1.0.zip 传到 /root/）：
#   sudo bash deploy_fast.sh
#   sudo bash deploy_fast.sh --port 8077 --zip /root/zhijian-agent-v1.0.zip
# =============================================================================
set -euo pipefail

PORT=8077
DIR="/opt/zhijian-agent"
ZIP=""
SERVICE=zhijian

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --dir)  DIR="$2";  shift 2 ;;
    --zip)  ZIP="$2";  shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "未知参数：$1"; exit 1 ;;
  esac
done

blue() { printf '\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "请用 root 或 sudo 执行"

blue "=============================================================="
blue " 智检 Agent · 快速部署（使用现成模型，不训练）"
blue "=============================================================="

# ---------------------------------------------------------------- 找安装包
if [[ -z "$ZIP" ]]; then
  for cand in /root/zhijian-agent-v1.0.zip /root/zhijian-agent-*.zip \
              /tmp/zhijian-agent-v1.0.zip ./zhijian-agent-v1.0.zip \
              "$(dirname "$0")/zhijian-agent-v1.0.zip"; do
    if [[ -f "$cand" ]]; then ZIP="$cand"; break; fi
  done
fi
[[ -n "$ZIP" && -f "$ZIP" ]] || die "找不到安装包。请把 zhijian-agent-v1.0.zip 传到 /root/ 后重跑"
echo "  安装包   : $ZIP ($(du -h "$ZIP" | cut -f1))"
echo "  安装目录 : $DIR"
echo "  服务端口 : $PORT"
echo

# ---------------------------------------------------------------- 1. 系统依赖
blue "[1/5] 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libgl1 / libglib2.0-0 是 opencv 的运行时依赖，缺了 import cv2 会报错
apt-get install -y -qq python3 python3-venv python3-pip unzip curl \
        libgl1 libglib2.0-0 libgomp1 >/dev/null
python3 -V
echo "  完成"

# ---------------------------------------------------------------- 2. 解压
blue "[2/5] 解压到 $DIR"
mkdir -p "$DIR"
STAGE=$(mktemp -d)
unzip -q -o "$ZIP" -d "$STAGE"
SRC="$STAGE/zhijian-agent"
[[ -d "$SRC" ]] || die "安装包结构异常，未找到 zhijian-agent/ 目录"
# 安装包里不含 data/app.db，这里是覆盖式合并，已有的工单数据不会被删
cp -a "$SRC/." "$DIR/"
rm -rf "$STAGE"
cd "$DIR"
mkdir -p runs/logs models data
echo "  文件数：$(find "$DIR" -type f | wc -l)"
[[ -f models/zhijian_best.pt ]] || die "安装包里没有模型权重 models/zhijian_best.pt"
[[ -f models/feature_bank.npz ]] || warn "  无开集特征库，未知缺陷拒识将自动关闭"

# ---------------------------------------------------------------- 3. 依赖
blue "[3/5] 安装 Python 依赖（CPU 版，约 200MB）"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
# extra-index-url 让 pip 同时看 PyPI 与 PyTorch CPU 源，
# 从而装上 +cpu 版本（比默认 CUDA 版小 10 倍）
pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
echo "  依赖安装完成"
python -c "import torch, cv2, fastapi; print('  torch', torch.__version__, '| cv2', cv2.__version__)"

# ---------------------------------------------------------------- 4. 口令 + 服务
blue "[4/5] 生成访问口令并注册 systemd 服务"
TOKEN_FILE="$DIR/.access_token"
if [[ -f "$TOKEN_FILE" ]]; then
  TOKEN=$(cat "$TOKEN_FILE")
  echo "  复用已有口令"
else
  TOKEN=$(head -c 12 /dev/urandom | base64 | tr -dc 'a-z0-9' | head -c 12)
  echo "$TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi

cat > /etc/systemd/system/${SERVICE}.service <<EOF
[Unit]
Description=Zhijian Agent (工业表面缺陷智能检测与处置系统)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$DIR
Environment="ZHJ_TOKEN=$TOKEN"
Environment="TORCH_HOME=$DIR/models/torch_cache"
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONUTF8=1"
ExecStart=$DIR/.venv/bin/python start.py --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=5
StandardOutput=append:$DIR/runs/logs/server.log
StandardError=append:$DIR/runs/logs/server.err.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE} >/dev/null 2>&1 || true
systemctl restart ${SERVICE} || true
sleep 5
if systemctl is-active --quiet ${SERVICE}; then
  echo "  服务已启动"
else
  warn "  服务未启动，日志如下："
  journalctl -u ${SERVICE} -n 30 --no-pager || true
  die "  启动失败"
fi

# 本机自测
if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
  echo "  本机健康检查通过"
else
  warn "  本机健康检查失败，请看：tail -50 runs/logs/server.err.log"
fi

# ---------------------------------------------------------------- 5. 防火墙
blue "[5/5] 放行防火墙端口 $PORT"
if command -v ufw >/dev/null 2>&1; then
  ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
  echo "  ufw 已放行"
fi
warn "  还需在云控制台【安全组/防火墙】放行 TCP $PORT，否则外网访问不到"

PUBLIC_IP=$(curl -fsS --max-time 6 https://api.ipify.org 2>/dev/null \
            || curl -fsS --max-time 6 https://ipinfo.io/ip 2>/dev/null \
            || echo "<服务器公网IP>")

blue "=============================================================="
blue " 部署完成"
blue "=============================================================="
cat <<EOF

  访问地址（发给评委）：
      http://${PUBLIC_IP}:${PORT}/?token=${TOKEN}

  访问口令：${TOKEN}

  自测：用手机流量（关 WiFi）打开上面地址，能出页面才算公网可达。

  常用命令：
      systemctl status ${SERVICE}
      systemctl restart ${SERVICE}
      journalctl -u ${SERVICE} -n 100
      tail -f $DIR/runs/logs/server.log

EOF
