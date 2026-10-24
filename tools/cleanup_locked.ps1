# ============================================================
# 清理 DSH 沙箱遗留的不可删除空目录
# ============================================================
# 背景：上一轮会话中，受限沙箱以 0o700 创建了一批临时目录。它们的
# 安全描述符只授予沙箱内部标识，当前用户既无读取、也无删除权限，
# 因此普通删不掉（icacls / takeown 也被拒绝）。本脚本需要在
# **管理员权限**下运行一次。
#
# 安全约束：
#   * 只处理下面 $Names 里明确列出的名字；
#   * 每个目标解析后的绝对路径必须位于 $Root 之内，否则跳过；
#   * 不做任何递归删除以外的事情，不碰项目文件。
#
# 用法（管理员 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File tools\cleanup_locked.ps1
# ============================================================

[CmdletBinding()]
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Continue'

$Names = @(
    # 工作区根目录下的 pytest 临时根
    '.pytest-basetemp'
    '.pytest-basetemp2'
    '.pytest-basetemp3'
    '.pytest-perm-probe'
    '.pytest-tmp'
    'pytest-of-StrFaith'
    'tmpn5ea1rco'
    '_sfp_probe_448'
    '_junk_sfp_probe_448'
    # 各子目录下的残留
    'tests\_pytmp'
    'tests\_sfp_probe_448'
    'outputs\_pytest'
    'outputs\_pytest_tmp'
    'outputs\_sfp_probe_448'
    'outputs\_pytestroot'
)

$RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
Write-Host "工作区: $RootFull"
Write-Host ""

$removed = 0
$skipped = 0
$failed = 0

# 通配匹配：pytest-cache-files-* 每次都不同名
$targets = @()
foreach ($name in $Names) {
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $RootFull $name))
    if (-not $candidate.StartsWith($RootFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "跳过（越界）: $name"
        $skipped++
        continue
    }
    $targets += $candidate
}
foreach ($dir in Get-ChildItem -LiteralPath $RootFull -Directory -Filter 'pytest-cache-files-*' -ErrorAction SilentlyContinue) {
    $targets += $dir.FullName
}

foreach ($path in $targets) {
    if (-not (Test-Path -LiteralPath $path)) { continue }
    try {
        # 先尝试接管所有权（管理员才有此特权），再显式授予删除权限
        & takeown.exe /F $path /D Y /R 2>&1 | Out-Null
        & icacls.exe $path /grant "*S-1-5-32-544:(OI)(CI)F" /T /C 2>&1 | Out-Null
        & icacls.exe $path /grant "$($env:USERNAME):(OI)(CI)F" /T /C 2>&1 | Out-Null
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        Write-Host "已删除: $($path.Substring($RootFull.Length + 1))"
        $removed++
    }
    catch {
        Write-Host "失败:   $($path.Substring($RootFull.Length + 1)) -> $($_.Exception.Message)"
        $failed++
    }
}

Write-Host ""
Write-Host "结果: 删除 $removed 个，失败 $failed 个，跳过 $skipped 个"
if ($failed -gt 0) {
    Write-Host "若仍失败，可在文件资源管理器中右键删除这些空目录。"
    exit 1
}
exit 0
