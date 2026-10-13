param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$toolsRoot = Join-Path $workspaceRoot '.tools'
$cacheRoot = Join-Path $toolsRoot 'downloads'
$lock = Get-Content -LiteralPath (Join-Path $workspaceRoot 'dependencies.lock.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null

foreach ($tool in $lock.windows_tools) {
    $target = Join-Path $toolsRoot $tool.directory
    $probe = switch ($tool.name) {
        'llvm-mingw' { Join-Path $target 'bin/clang++.exe' }
        'cmake' { Join-Path $target 'bin/cmake.exe' }
        'ninja' { Join-Path $target 'ninja.exe' }
        'python' { Join-Path $target 'python.exe' }
        default { throw ('不支持的工具：' + $tool.name) }
    }
    if (Test-Path -LiteralPath $probe) {
        Write-Host ('已安装：' + $tool.name + ' ' + $tool.version)
        continue
    }
    $archive = Join-Path $cacheRoot ([System.IO.Path]::GetFileName(([uri]$tool.url).AbsolutePath))
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host ('下载：' + $tool.name + ' ' + $tool.version)
        Invoke-WebRequest -Uri $tool.url -OutFile ($archive + '.partial')
        Move-Item -LiteralPath ($archive + '.partial') -Destination $archive -Force
    }
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $tool.sha256) { throw ('校验失败，停止安装：' + $archive) }
    $destination = if ($tool.name -in @('ninja', 'python')) { $target } else { $toolsRoot }
    Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force
    if (-not (Test-Path -LiteralPath $probe)) { throw ('解压后未找到工具：' + $probe) }
    Write-Host ('安装完成：' + $tool.name + ' ' + $tool.version)
}

Write-Host '便携工具已安装在仓库 .tools 中；未修改系统 PATH。'
