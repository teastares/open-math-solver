#!/usr/bin/env python3
"""按固定哈希准备现代公开 LP 开发冒烟输入，不求解模型。"""

import argparse
import bz2
import hashlib
import json
from pathlib import Path
import re
import ssl
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
SOURCE = "https://plato.asu.edu/ftp/lptestset/"
COMPRESSED_LIMIT = 500_000_000
EXPANDED_LIMIT = 1_000_000_000
CHUNK = 1024 * 1024


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def verify_identity(identity, expected_bytes, expected_sha256):
    if expected_bytes is not None and identity["bytes"] != expected_bytes:
        raise ValueError("文件字节数与锁不一致")
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError("文件 SHA256 与锁不一致")


def safe_target(root, name):
    target = root / name
    if target.resolve().parent != root:
        raise ValueError("文件路径超出输出目录：" + name)
    return target


def validate_lock(lock):
    if not isinstance(lock, dict) or type(lock.get("schema_version")) is not int or lock["schema_version"] != 1 or lock.get("role") != "development_smoke":
        raise ValueError("锁必须明确标记为版本 1 的开发冒烟输入")
    models = lock.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("锁内模型列表不能为空")
    names = set()
    compressed = expanded = 0
    reserved = {"CON", "PRN", "AUX", "NUL"} | {prefix + str(i) for prefix in ("COM", "LPT") for i in range(1, 10)}
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("模型锁条目必须是对象")
        name = model.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.mps", name) or name.split(".")[0].upper() in reserved:
            raise ValueError("非法模型文件名")
        if name.casefold() in names:
            raise ValueError("锁内模型文件名重复")
        names.add(name.casefold())
        if model.get("url") != SOURCE + name + ".bz2":
            raise ValueError("数据 URL 必须精确对应指定官方目录与文件名")
        for field in ("compressed_bytes", "bytes"):
            if type(model.get(field)) is not int or model[field] <= 0:
                raise ValueError("锁定文件字节数必须是正整数")
        for field in ("compressed_sha256", "sha256"):
            if not isinstance(model.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", model[field]):
                raise ValueError("锁缺少合法 SHA256")
        compressed += model["compressed_bytes"]
        expanded += model["bytes"]
    if compressed > COMPRESSED_LIMIT or expanded > EXPANDED_LIMIT:
        raise ValueError("完整清单超出 500 MB 压缩或 1 GB 解压预算，未下载任何数据")
    return compressed, expanded


def publish(temporary, target, identity):
    # 独占发布，既有不同内容和并发写入均不覆盖。
    try:
        target.hardlink_to(temporary)
        return True
    except FileExistsError:
        verify_identity(file_identity(target), identity["bytes"], identity["sha256"])
        return False


def download(url, target, byte_limit, expected_bytes=None, expected_sha256=None,
             context=None, opener=None):
    opener = opener or urllib.request.urlopen
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False) as output:
        temporary = Path(output.name)
        try:
            digest = hashlib.sha256()
            size = 0
            with opener(url, context=context, timeout=60) as response:
                if response.geturl() != url:
                    raise ValueError("官方数据地址发生重定向，需重新核对来源后冻结")
                length = response.headers.get("Content-Length")
                if length is not None and (int(length) > byte_limit or
                                           (expected_bytes is not None and int(length) != expected_bytes)):
                    raise ValueError("服务器报告的文件体积与预算或锁不一致")
                while chunk := response.read(min(CHUNK, byte_limit - size + 1)):
                    size += len(chunk)
                    if size > byte_limit:
                        raise ValueError("实际下载超出压缩文件预算")
                    output.write(chunk)
                    digest.update(chunk)
            identity = {"bytes": size, "sha256": digest.hexdigest()}
            verify_identity(identity, expected_bytes, expected_sha256)
            output.close()
            created = publish(temporary, target, identity)
            return identity, created
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def expand(archive, target, byte_limit, expected_bytes=None, expected_sha256=None):
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False) as output:
        temporary = Path(output.name)
        try:
            decoder = bz2.BZ2Decompressor()
            digest = hashlib.sha256()
            size = 0
            with archive.open("rb") as source:
                while chunk := source.read(CHUNK):
                    if decoder.eof:
                        raise ValueError("bz2 结束后存在额外字节")
                    while True:
                        data = decoder.decompress(chunk, max_length=min(CHUNK, byte_limit - size + 1))
                        chunk = b""
                        size += len(data)
                        if size > byte_limit:
                            raise ValueError("实际解压超出模型文件预算")
                        output.write(data)
                        digest.update(data)
                        if decoder.eof:
                            if decoder.unused_data:
                                raise ValueError("bz2 含有尾随数据或额外压缩流")
                            break
                        if decoder.needs_input:
                            break
            if not decoder.eof:
                raise ValueError("bz2 文件不完整")
            identity = {"bytes": size, "sha256": digest.hexdigest()}
            verify_identity(identity, expected_bytes, expected_sha256)
            output.close()
            created = publish(temporary, target, identity)
            return identity, created
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def fetch(lock_path, output_dir, ca_file=None, opener=None):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    compressed, expanded = validate_lock(lock)
    output_root = output_dir.resolve()
    archive_root = output_root / ".archives"
    if archive_root.resolve().parent != output_root:
        raise ValueError("归档缓存目录超出输出目录")
    targets = [(model, safe_target(output_root, model["name"]),
                safe_target(archive_root.resolve(), model["name"] + ".bz2")) for model in lock["models"]]
    # 全部路径与现有文件先核对，避免后续才发现用户文件将被覆盖。
    for model, target, archive in targets:
        if target.exists():
            verify_identity(file_identity(target), model["bytes"], model["sha256"])
        if archive.exists():
            verify_identity(file_identity(archive), model["compressed_bytes"], model["compressed_sha256"])
    archive_root.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
    new_files = new_archives = 0
    for model, target, archive in targets:
        if not archive.exists():
            _, created = download(model["url"], archive, model["compressed_bytes"],
                                  model["compressed_bytes"], model["compressed_sha256"], context, opener)
            new_archives += created
        _, created = expand(archive, target, model["bytes"], model["bytes"], model["sha256"])
        new_files += created
    return {"model_count": len(targets), "new_files": new_files, "new_archives": new_archives,
            "compressed_bytes": compressed, "expanded_bytes": expanded, "output_dir": str(output_root),
            "role": "development_smoke", "skipped": lock.get("skipped", [])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "benchmarks" / "modern-lp-smoke.lock.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "data" / "modern-lp-smoke")
    parser.add_argument("--ca-file", type=Path, help="可选可信 CA PEM 文件；始终启用证书验证")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(fetch(args.lock, args.output_dir, args.ca_file), ensure_ascii=False))
        return 0
    except (OSError, ValueError) as error:
        print("准备现代 LP 失败：" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
