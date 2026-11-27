"""将已构建的 macOS x86_64 Ninja Release 程序重链至系统 Accelerate。"""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ninja", default="ninja")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Accelerate 仅在 macOS 可用")
    build = args.build.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("输出父目录必须存在，且输出文件不能已存在")
    binary = build / "solver"
    arch = subprocess.check_output(["/usr/bin/lipo", "-archs", str(binary)], text=True).strip()
    if arch != "x86_64":
        parser.error("需要已构建的 x86_64 CLI")
    commands = subprocess.check_output([args.ninja, "-t", "commands", "solver"], cwd=build, text=True)
    link = shlex.split(commands.splitlines()[-1])
    if link[:2] != [":", "&&"] or link[-2:] != ["&&", ":"]:
        parser.error("无法识别 Ninja 链接命令")
    link = link[2:-2]
    if "-o" not in link or not any(Path(arg).name == "libopenblas.a" for arg in link):
        parser.error("链接命令须使用静态 OpenBLAS")
    link[link.index("-o") + 1] = str(output)
    link = [arg for arg in link if Path(arg).name != "libopenblas.a"]
    subprocess.run([*link, "-framework", "Accelerate"], cwd=build, check=True)
    subprocess.run([str(output), "--version"], check=True)


if __name__ == "__main__":
    main()
