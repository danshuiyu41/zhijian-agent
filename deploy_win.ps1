# =============================================================================
# 智检 Agent · 从 Windows 一键部署到云服务器
# -----------------------------------------------------------------------------
# 做三件事：传安装包 -> 传部署脚本 -> 远程执行部署
#
# 用法（在项目根目录打开 PowerShell）：
#   .\deploy_win.ps1 -Server 1.2.3.4
#   .\deploy_win.ps1 -Server 1.2.3.4 -Port 8077
#
# 会提示输两次服务器密码（第一次传文件，第二次执行部署）。
# =============================================================================
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [string]$User = "root",
    [int]$Port = 8077,
    [string]$Zip = ""
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

function Info($m) { Write-Host $m -ForegroundColor Cyan }
function Good($m) { Write-Host $m -ForegroundColor Green }
function Warn($m) { Write-Host $m -ForegroundColor Yellow }
function Fail($m) { Write-Host $m -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host (" 智检 Agent · 部署到 " + $Server) -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------ 0. 前置检查
Info "[0/3] 检查本机工具链"

foreach ($tool in @("ssh", "scp")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "未找到 $tool。请先启用 Windows 自带的 OpenSSH 客户端：" -ForegroundColor Red
        Write-Host "  设置 -> 应用 -> 可选功能 -> 添加功能 -> 搜索 OpenSSH 客户端 -> 安装" -ForegroundColor Red
        Write-Host "装完重开 PowerShell 再跑本脚本。" -ForegroundColor Red
        exit 1
    }
}
Good "  ssh / scp 可用"

if (-not $Zip) {
    $cand = Join-Path $Root "dist\zhijian-agent-v1.0.zip"
    if (Test-Path $cand) { $Zip = $cand }
}
if (-not (Test-Path $Zip)) {
    Fail ("找不到安装包：" + $Zip + "；请先运行 python -m src.make_installer")
}
$zipMB = [math]::Round((Get-Item $Zip).Length / 1MB, 2)
Good ("  安装包：" + $Zip + "（" + $zipMB + " MB）")

$sh = Join-Path $Root "deploy_fast.sh"
if (-not (Test-Path $sh)) { Fail ("找不到部署脚本：" + $sh) }
Good ("  部署脚本：" + $sh)

Info ("  探测 " + $Server + ":22 ...")
$tcp = Test-NetConnection -ComputerName $Server -Port 22 -WarningAction SilentlyContinue
if (-not $tcp.TcpTestSucceeded) {
    Write-Host ("连不上 " + $Server + ":22。可能原因：") -ForegroundColor Red
    Write-Host "  (1) IP 填错了" -ForegroundColor Red
    Write-Host "  (2) 云控制台安全组没放行 22 端口" -ForegroundColor Red
    Write-Host "  (3) 服务器还没启动完成（买完等 1~2 分钟）" -ForegroundColor Red
    exit 1
}
Good "  22 端口可达"

# ------------------------------------------------ 1. 传文件
Write-Host ""
Info "[1/3] 上传安装包与部署脚本（会提示输入服务器密码）"
Write-Host ""

& scp -o StrictHostKeyChecking=accept-new $Zip ($User + "@" + $Server + ":/root/zhijian-agent-v1.0.zip")
if ($LASTEXITCODE -ne 0) { Fail "上传安装包失败（密码错？磁盘满？）" }

& scp -o StrictHostKeyChecking=accept-new $sh ($User + "@" + $Server + ":/root/deploy_fast.sh")
if ($LASTEXITCODE -ne 0) { Fail "上传部署脚本失败" }

Good "  上传完成"

# ------------------------------------------------ 2. 远程部署
Write-Host ""
Info "[2/3] 在服务器上执行部署（约 3~5 分钟，会再次提示输密码）"
Write-Host ""

$remote = "sed -i 's/\r$//' /root/deploy_fast.sh && bash /root/deploy_fast.sh --port $Port"
& ssh -o StrictHostKeyChecking=accept-new ($User + "@" + $Server) $remote
if ($LASTEXITCODE -ne 0) { Fail "远程部署失败，请看上面输出的报错" }

# ------------------------------------------------ 3. 收尾
Write-Host ""
Info "[3/3] 部署完成"
Write-Host ""
Write-Host "--------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "还有两件必须做的事，否则外网打不开：" -ForegroundColor Yellow
Write-Host ""
Write-Host ("  1. 去云控制台 -> 你的服务器 -> 安全组/防火墙") -ForegroundColor Yellow
Write-Host ("     添加规则：入方向、TCP、端口 " + $Port + "、来源 0.0.0.0/0") -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. 用【手机流量】（关掉 WiFi）打开服务器打印出的地址自测" -ForegroundColor Yellow
Write-Host ""
Write-Host "想拿到访问口令：" -ForegroundColor Yellow
Write-Host ("     ssh " + $User + "@" + $Server + " `"cat /opt/zhijian-agent/.access_token`"") -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------" -ForegroundColor Yellow
