#!/usr/bin/env bash
# =============================================================================
# 智检 Agent · 云服务器一键部署脚本（Ubuntu / Debian）
# -----------------------------------------------------------------------------
# 做的事：
#   1. 安装系统依赖与 Python 包（CPU 版 PyTorch，无需显卡）
#   2. 拉取代码（仓库不含数据集与权重）
#   3. 重建数据集与模型（合成数据 → 训练 → 指标 → 看板数据）
#   4. 生成访问口令，写入配置
#   5. 注册 systemd 服务（开机自启、崩溃自动重启）
#   6. 放行防火墙端口
#
# 用法（在云服务器上执行）：
#   curl -fsSL <本脚本的公网地址> | bash -s -- --port 8077
#   或上传后执行：  chmod +x deploy.sh && sudo ./deploy.sh
#
#   ./deploy.sh --port 8077 --repo git@github.com:danshuiyu41/zhijian-agent.git
#   ./deploy.sh --skip-train     # 跳过训练（复用已有权重，用于快速重启部署）
# =============================================================================
set -euo pipefail

PORT=8077
REPO="git@github.com:danshuiyu41/zhijian-agent.git"
DIR="/opt/zhijian-agent"
SKIP_TRAIN=0
PER_CLASS=200
EPOCHS=18
SERVICE=zhijian

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --dir)  DIR="$2";  shift 2 ;;
    --per-class) PER_CLASS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "未知参数：$1"; exit 1 ;;
  esac
done

blue() { printf '\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "请用 root 或 sudo 执行本脚本"

blue "=============================================================="
blue " 智检 Agent · 云服务器部署"
blue "=============================================================="
echo "  安装目录 : $DIR"
echo "  服务端口 : $PORT"
echo "  代码仓库 : $REPO"
echo

# ---------------------------------------------------------------- 1. 系统依赖
blue "[1/6] 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl ufw >/dev/null
python3 -V

# ---------------------------------------------------------------- 2. 拉代码
blue "[2/6] 获取代码"
if [[ -d "$DIR/.git" ]]; then
  echo "  已存在，执行 git pull"
  git -C "$DIR" pull --ff-only || warn "  pull 失败（本地有改动？），继续使用现有代码"
else
  if ! git clone "$REPO" "$DIR"; then
    warn "  SSH 方式拉取失败。两种解决办法："
    warn "   ① 在服务器生成密钥并加到 GitHub：ssh-keygen -t ed25519 后把 ~/.ssh/id_ed25519.pub 加到 GitHub"
    warn "   ② 改用 HTTPS 地址：--repo https://github.com/用户名/仓库.git"
    die  "  代码获取失败，请处理后重跑"
  fi
fi
cd "$DIR"

# ---------------------------------------------------------------- 3. 依赖
blue "[3/6] 安装 Python 依赖（CPU 版，无需显卡）"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
# PyTorch CPU 专用源：体积比默认 CUDA 版小很多（约 200MB vs 2.5GB）
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q opencv-python-headless numpy fastapi uvicorn python-multipart matplotlib pyyaml pillow

# 内存检查：训练很吃内存，低于 3GB 会失败
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "  可用内存：${MEM_MB} MB"
if [[ "$MEM_MB" -lt 3000 ]]; then
  warn "  内存偏低（<3GB），训练可能被 OOM 杀掉。"
  warn "  建议：加 swap，或改用 --skip-train 并手动上传权重文件"
fi

# ---------------------------------------------------------------- 4. 数据与模型
blue "[4/6] 重建数据集与模型"
export TORCH_HOME="$DIR/models/torch_cache"
mkdir -p "$TORCH_HOME" models runs data

if [[ "$SKIP_TRAIN" -eq 1 ]]; then
  warn "  跳过训练。请自行把 models/*.pt 与 models/*.npz 上传到 $DIR/models/"
else
  echo "  → 生成合成数据集（每类 $PER_CLASS 张）"
  python -m src.data_generate --per-class "$PER_CLASS" --stress --stress-per-class 30
  echo "  → 构建数据索引"
  python -m src.data
  echo "  → 训练模型（$EPOCHS 轮，CPU 约 30–50 分钟，请耐心等待）"
  python -m src.train --epochs "$EPOCHS" --batch-size 32
  echo "  → 生成指标文档"
  python -m src.make_metrics || true
  echo "  → 建立未知缺陷识别特征库"
  python -m src.ood --fit && python -m src.ood --eval || warn "  特征库构建失败，未知缺陷识别将自动降级关闭"
fi

# ---------------------------------------------------------------- 5. 访问口令与服务
blue "[5/6] 生成访问口令并注册服务"
TOKEN_FILE="$DIR/.access_token"
if [[ -f "$TOKEN_FILE" ]]; then
  TOKEN=$(cat "$TOKEN_FILE")
  echo "  复用已有口令"
else
  TOKEN=$(head -c 9 /dev/urandom | base64 | tr -d '/+=' | head -c 10)
  echo "$TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi

# 允许开机自启、并让服务以明文 HTTP 对外（演示环境；如需 HTTPS 请自行加 Nginx）
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

mkdir -p "$DIR/runs/logs"
systemctl daemon-reload
systemctl enable --now ${SERVICE}
sleep 3
systemctl is-active --quiet ${SERVICE} && echo "  服务已启动" || warn "  服务未正常启动，请查看：journalctl -u ${SERVICE} -n 50"

# ---------------------------------------------------------------- 6. 防火墙
blue "[6/6] 放行防火墙端口 $PORT"
if command -v ufw >/dev/null 2>&1; then
  ufw allow "${PORT}/tcp" >/dev/null 2>&1 || true
  echo "  ufw 已放行（若 ufw 未启用则无需处理）"
fi
warn "  别忘了在云控制台【安全组】里也放行 TCP $PORT —— 这一步不做，外网访问不到"

# ---------------------------------------------------------------- 完成
PUBLIC_IP=$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo "<服务器公网IP>")
blue "=============================================================="
blue " 部署完成"
blue "=============================================================="
cat <<EOF

  访问地址（给评委）：
      http://${PUBLIC_IP}:${PORT}/?token=${TOKEN}

  访问口令：${TOKEN}
      （已写入 $TOKEN_FILE，可在配置文件 config.yaml 的 server.access_token 修改）

  常用命令：
      systemctl status ${SERVICE}      查看状态
      systemctl restart ${SERVICE}     重启
      systemctl stop ${SERVICE}        停止
      journalctl -u ${SERVICE} -n 100  查看日志
      tail -f $DIR/runs/logs/server.log

  自测（关键）：
      用**手机流量**打开上面的地址，能出页面才算真正公网可达。
      若打不开，依次检查：
        ① 云控制台安全组是否放行 TCP $PORT
        ② ufw 状态：ufw status
        ③ 服务是否在跑：systemctl is-active ${SERVICE}
        ④ 端口是否监听：ss -lntp | grep $PORT

EOF
