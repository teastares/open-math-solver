param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Configuration = 'Release',
    [int]$Jobs = 4,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
if ($Jobs -lt 1) { throw 'Jobs 必须为正整数。' }
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$lock = Get-Content -LiteralPath (Join-Path $workspaceRoot 'dependencies.lock.json') -Raw | ConvertFrom-Json
$compiler = Join-Path $workspaceRoot ('.tools/' + ($lock.windows_tools | Where-Object name -eq 'llvm-mingw').directory + '/bin')
$cmake = Join-Path $workspaceRoot ('.tools/' + ($lock.windows_tools | Where-Object name -eq 'cmake').directory + '/bin')
$ninja = Join-Path $workspaceRoot ('.tools/' + ($lock.windows_tools | Where-Object name -eq 'ninja').directory)
$python = Join-Path $workspaceRoot ('.tools/' + ($lock.windows_tools | Where-Object name -eq 'python').directory + '/python.exe')
if (-not (Test-Path -LiteralPath (Join-Path $compiler 'clang++.exe'))) {
    throw '请先运行 scripts/bootstrap-windows.ps1 安装便携工具链。'
}
$previousPath = $env:PATH
try {
    $env:PATH = "$compiler;$cmake;$ninja;$previousPath"
    $blasBuild = Join-Path $workspaceRoot 'build/openblas'
    & (Join-Path $cmake 'cmake.exe') -S (Join-Path $workspaceRoot 'third_party/openblas') -B $blasBuild -G Ninja '-DCMAKE_C_COMPILER=clang' "-DSOLVER_DEPENDENCY_JOBS=$Jobs"
    if ($LASTEXITCODE -ne 0) { throw 'OpenBLAS 依赖配置失败。' }
    & (Join-Path $cmake 'cmake.exe') --build $blasBuild --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'OpenBLAS 依赖构建失败。' }
    $build = Join-Path $workspaceRoot ('build/' + $Configuration.ToLowerInvariant())
    & (Join-Path $cmake 'cmake.exe') -S $workspaceRoot -B $build -G Ninja "-DCMAKE_BUILD_TYPE=$Configuration" '-DCMAKE_C_COMPILER=clang' '-DCMAKE_CXX_COMPILER=clang++' '-DCMAKE_EXE_LINKER_FLAGS=-static' '-DSOLVER_BUILD_TESTS=ON' "-DPython3_EXECUTABLE=$python"
    if ($LASTEXITCODE -ne 0) { throw 'CMake 配置失败。' }
    & (Join-Path $cmake 'cmake.exe') --build $build --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw '编译失败。' }
    if (-not $SkipTests) {
        & (Join-Path $cmake 'ctest.exe') --test-dir $build --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw '回归测试失败。' }
    }
} finally {
    $env:PATH = $previousPath
}
