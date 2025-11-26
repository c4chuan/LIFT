# 轨迹推理填充系统 - 启动脚本 (PowerShell)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "轨迹推理填充系统" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "检查 Python 环境..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到 Python" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion" -ForegroundColor Green
Write-Host ""

# 检查依赖
Write-Host "检查依赖..." -ForegroundColor Yellow
$packages = @("dashscope", "pillow", "pyyaml")
$missingPackages = @()

foreach ($package in $packages) {
    $result = python -c "import $package" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "缺少以下依赖包: $($missingPackages -join ', ')" -ForegroundColor Red
    Write-Host "安装依赖? (Y/N): " -NoNewline
    $response = Read-Host
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host "安装中..." -ForegroundColor Yellow
        pip install $($missingPackages -join ' ')
        if ($LASTEXITCODE -ne 0) {
            Write-Host "安装失败" -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ 依赖安装完成" -ForegroundColor Green
    } else {
        Write-Host "请手动安装依赖: pip install $($missingPackages -join ' ')" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✓ 所有依赖已安装" -ForegroundColor Green
}
Write-Host ""

# 检查 API Key
Write-Host "检查 API Key..." -ForegroundColor Yellow
$apiKey = $env:DASHSCOPE_API_KEY

if (-not $apiKey) {
    Write-Host "未设置环境变量 DASHSCOPE_API_KEY" -ForegroundColor Yellow
    Write-Host "请输入您的 DashScope API Key: " -NoNewline
    $apiKey = Read-Host
    if (-not $apiKey) {
        Write-Host "错误: API Key 不能为空" -ForegroundColor Red
        exit 1
    }
    $env:DASHSCOPE_API_KEY = $apiKey
    Write-Host "✓ API Key 已设置（仅本次会话）" -ForegroundColor Green
} else {
    Write-Host "✓ API Key 已设置" -ForegroundColor Green
}
Write-Host ""

# 显示菜单
Write-Host "请选择操作:" -ForegroundColor Cyan
Write-Host "1. 运行测试（不调用 API）" -ForegroundColor White
Write-Host "2. 运行测试（调用 API）" -ForegroundColor White
Write-Host "3. 处理单个环境（classifieds）" -ForegroundColor White
Write-Host "4. 处理前 5 个轨迹（测试）" -ForegroundColor White
Write-Host "5. 批量处理所有轨迹" -ForegroundColor White
Write-Host "6. 查看进度" -ForegroundColor White
Write-Host "7. 重置进度" -ForegroundColor White
Write-Host "0. 退出" -ForegroundColor White
Write-Host ""
Write-Host "请输入选项 (0-7): " -NoNewline -ForegroundColor Yellow
$choice = Read-Host

Write-Host ""

switch ($choice) {
    "1" {
        Write-Host "运行测试（Dry Run）..." -ForegroundColor Cyan
        python -m src.reasoning_filler.test_single
    }
    "2" {
        Write-Host "运行测试（实际 API 调用）..." -ForegroundColor Cyan
        python -m src.reasoning_filler.test_single --no-dry-run
    }
    "3" {
        Write-Host "处理 classifieds 环境..." -ForegroundColor Cyan
        python -m src.reasoning_filler.main --env classifieds
    }
    "4" {
        Write-Host "处理前 5 个轨迹..." -ForegroundColor Cyan
        python -m src.reasoning_filler.main --max_count 5
    }
    "5" {
        Write-Host "批量处理所有轨迹..." -ForegroundColor Cyan
        Write-Host "警告: 这将处理所有轨迹文件，可能需要较长时间" -ForegroundColor Yellow
        Write-Host "确认继续? (Y/N): " -NoNewline
        $confirm = Read-Host
        if ($confirm -eq 'Y' -or $confirm -eq 'y') {
            python -m src.reasoning_filler.main
        } else {
            Write-Host "已取消" -ForegroundColor Yellow
        }
    }
    "6" {
        Write-Host "查看进度..." -ForegroundColor Cyan
        if (Test-Path "data/annotate_with_reasoning/progress.json") {
            Get-Content "data/annotate_with_reasoning/progress.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
        } else {
            Write-Host "未找到进度文件" -ForegroundColor Yellow
        }
    }
    "7" {
        Write-Host "重置进度..." -ForegroundColor Cyan
        Write-Host "警告: 这将删除所有进度记录" -ForegroundColor Yellow
        Write-Host "确认继续? (Y/N): " -NoNewline
        $confirm = Read-Host
        if ($confirm -eq 'Y' -or $confirm -eq 'y') {
            python -m src.reasoning_filler.main --reset_progress --max_count 0
            Write-Host "✓ 进度已重置" -ForegroundColor Green
        } else {
            Write-Host "已取消" -ForegroundColor Yellow
        }
    }
    "0" {
        Write-Host "退出" -ForegroundColor Cyan
        exit 0
    }
    default {
        Write-Host "无效选项" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "完成!" -ForegroundColor Green
