#!/usr/bin/env python3
"""按固定归档与模型哈希准备公开 Netlib 输入，不执行归档中的代码。"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent.parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="使用已经下载的归档")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".deps" / "netlib")
    args = parser.parse_args()
    lock = json.loads((ROOT / "benchmarks" / "netlib.lock.json").read_text(encoding="utf-8"))
    archive = args.archive or ROOT / ".deps" / ("netlib-" + lock["revision"] + ".tar.gz")
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=archive.name + ".",
                                         suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            try:
                with urllib.request.urlopen(lock["url"], timeout=60) as response:
                    shutil.copyfileobj(response, output)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            if digest(temporary.read_bytes()) != lock["sha256"]:
                raise ValueError("下载归档的 SHA256 不一致，停止准备数据")
            try:
                # 硬链接发布是独占创建；其他进程已发布时只核对内容。
                archive.hardlink_to(temporary)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    if digest(archive.read_bytes()) != lock["sha256"]:
        raise ValueError("现有归档的 SHA256 不一致，停止准备数据")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(archive, "r:gz") as source:
        candidates = {}
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if member.isfile() and len(path.parts) == 2 and path.name.endswith(".mps.gz"):
                name = path.name[:-3]
                if name in candidates:
                    raise ValueError("归档包含重复的模型文件：" + name)
                candidates[name] = member
        for model in lock["models"]:
            name = model["name"]
            target = output_root / name
            if target.resolve().parent != output_root or not name.endswith(".mps"):
                raise ValueError("模型文件名超出目标目录")
            if name not in candidates:
                raise ValueError("归档缺少锁定模型：" + name)
            with source.extractfile(candidates[name]) as compressed:
                data = gzip.decompress(compressed.read())
            if digest(data) != model["sha256"] or len(data) != model["bytes"]:
                raise ValueError("模型校验失败：" + name)
            if target.exists():
                if digest(target.read_bytes()) != model["sha256"]:
                    raise ValueError("已有模型与锁定数据不同，拒绝覆盖：" + name)
            else:
                with target.open("xb") as output:
                    output.write(data)
                extracted += 1
    print(json.dumps({"model_count": len(lock["models"]), "new_files": extracted,
                      "output_dir": str(output_root), "revision": lock["revision"]}))


if __name__ == "__main__":
    main()
