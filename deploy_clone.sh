#!/usr/bin/env bash
# =============================================================================
# 智检 Agent · 仓库已 clone 后的服务器配置脚本
# -----------------------------------------------------------------------------
# 适用场景：你已经在服务器上 git clone 了仓库，想把环境装好、跑起来。
#
# 前提：把 3 个模型文件从你自己的电脑传到服务器的 models/ 目录（仓库里不含权重）：
#   在 Windows 上执行（不是服务器上）：
#   scp D:\deeps\bisai\models\zhijian_best.pt `
#       D:\deeps\bisai\models\feature_bank.npz `
#       D:\deeps\bisai\models\ood_config.json `
#       root@服务器IP:/root/zhijian-agent/models/
#
# 然后在服务器上、仓库根目录执行：
#   sudo bash deploy_clone.sh
#   sudo bash deploy_clone.sh --port 8077
# =============================================================================
set -euo pipefail

PORT=8077
SERVICE=zhijian
DIR="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "未知参数：$1"; exit 1 ;;
  esac
done

blue() { printf '\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "请用 sudo 执行"
[[ -f "$DIR/src/api.py" ]] || die "$DIR 不是仓库根目录（找不到 src/api.py）"

blue "=============================================================="
blue " 智检 Agent · 环境安装（已 clone 仓库）"
blue "=============================================================="
echo "  仓库目录 : $DIR"
echo "  服务端口 : $PORT"
echo

# ---------------------------------------------------------------- 0. 权重检查
blue "[0/5] 检查模型权重"
MISSING=0
for f in models/zhijian_best.pt models/feature_bank.npz models/ood_config.json; do
  if [[ -f "$DIR/$f" ]]; then
    echo "  [OK]   $f ($(du -h "$DIR/$f" | cut -f1))"
  else
    echo "  [缺失] $f"
    MISSING=1
  fi
done
if [[ "$MISSING" -eq 1 ]]; then
  cat >&2 <<'EOF'

仓库里不含模型权重（体积与可复现性考虑）。两种解决办法，选一个：

  A. 从自己的电脑上传（推荐，1 分钟）
     在你 Windows 的 PowerShell 里执行：
       scp D:\deeps\bisai\models\zhijian_best.pt `
           D:\deeps\bisai\models\feature_bank.npz `
           D:\deeps\bisai\models\ood_config.json `
           root@服务器IP:仓库路径/models/
     然后重跑本脚本。

  B. 在服务器上重新训练（约 40 分钟，需 ≥4GB 内存）
       sudo bash deploy.sh --skip-train 0
     或手动：
       python3 -m venv .venv && source .venv/bin/activate
       pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
       python -m src.data_generate --per-class 200 --stress --stress-per-class 30
       python -m src.data && python -m src.train --epochs 18 --batch-size 32
       python -m src.ood --fit

EOF
  die "缺少模型权重，请先按上面任一方式补齐"
fi

# ---------------------------------------------------------------- 1. 系统依赖
blue "[1/5] 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# libgl1 / libglib2.0-0 是 opencv 的运行时依赖；libgomp1 是 torch 的
apt-get install -y -qq python3 python3-venv python3-pip curl \
        libgl1 libglib2.0-0 libgomp1 >/dev/null
python3 -V

MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "  内存：${MEM_MB} MB"
if [[ "$MEM_MB" -lt 1500 ]]; then
  warn "  内存不足 1.5GB，推理可能不稳。建议加 2G swap："
  warn "  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile"
fi

# ---------------------------------------------------------------- 2. Python 依赖
blue "[2/5] 安装 Python 依赖（CPU 版 PyTorch，约 200MB）"
cd "$DIR"
[[ -d .venv ]] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip

PYV=$(python3 -c 'import sys;print(sys.version_info.major*100+sys.version_info.minor)')
echo "  Python: $(python3 -V 2>&1)  (开发环境为 3.10)"
if [[ "$PYV" -ge 312 ]]; then
  warn "  Python ≥ 3.12：requirements.txt 里冻结的老版本 numpy 没有预编译包，"
  warn "  改用不固定版本的方式安装（功能等价，仅版本更新）"
  pip install -q torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
  pip install -q "numpy>=1.26" opencv-python-headless fastapi uvicorn \
                 python-multipart PyYAML Pillow requests
else
  pip install -q -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
fi
python -c "import torch, cv2, fastapi; print('  torch', torch.__version__, '| cv2', cv2.__version__)"

# ---------------------------------------------------------------- 3. 启动自检
blue "[3/5] 启动前自检"
export TORCH_HOME="$DIR/models/torch_cache"
mkdir -p "$TORCH_HOME" runs/logs data
python start.py --check || warn "  自检有告警，但可以继续（无数据集不影响推理）"

# ---------------------------------------------------------------- 4. 口令 + 服务
blue "[4/5] 生成访问口令并注册 systemd 服务"
TOKEN_FILE="$DIR/.access_token"
if [[ -f "$TOKEN_FILE" ]]; then
  TOKEN=$(cat "$TOKEN_FILE"); echo "  复用已有口令"
else
  TOKEN=$(head -c 12 /dev/urandom | base64 | tr -dc 'a-z0-9' | head -c 12)
  echo "$TOKEN" > "$TOKEN_FILE"; chmod 600 "$TOKEN_FILE"
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
sleep 6

if systemctl is-active --quiet ${SERVICE}; then
  echo "  服务已启动"
else
  warn "  服务未启动，最近 30 行日志："
  journalctl -u ${SERVICE} -n 30 --no-pager || true
  die "  启动失败"
fi

if curl -fsS --max-time 8 "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
  echo "  本机健康检查通过"
else
  warn "  本机健康检查失败，看日志：tail -50 $DIR/runs/logs/server.err.log"
fi

# ---------------------------------------------------------------- 5. 防火墙
blue "[5/5] 放行防火墙端口 $PORT"
if command -v ufw >/dev/null 2>&1; then
  ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
  echo "  ufw 已放行"
fi
warn "  别忘了云控制台【安全组】放行 TCP $PORT，否则外网访问不到"

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
  打不开就依次查：安全组是否放行 $PORT → systemctl status $SERVICE
                  → ss -lntp | grep $PORT

  常用命令：
      systemctl status ${SERVICE}
      systemctl restart ${SERVICE}
      journalctl -u ${SERVICE} -n 100

EOF
