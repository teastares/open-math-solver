"""归档第46轮的独立诊断来源、构建记录和快照，不运行求解器。"""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = DATA / "v49-diagnostic-artifacts.zip"
    manifest = DATA / "v49-archive-manifest.json"
    assert not archive.exists() and not manifest.exists()
    run = json.loads((DATA / "v49-run-manifest.json").read_text(encoding="utf-8"))
    work = Path(run["work"])
    files = {}

    def add(path, name):
        assert name not in files and path.is_file()
        files[name] = path

    # 冻结完整源码沿用v48包；这里仅保存新增来源和两份转换结果。
    for pattern in ("v49-*.py", "v49-*.hpp", "v49-*.cpp", "v49-plan.json", "v49-diagnostic-clarification.json"):
        for path in sorted(DATA.glob(pattern)):
            add(path, "sources/" + path.name)
    for path in sorted(work.iterdir()):
        if path.is_file() and path.suffix in (".json", ".bin", ".cpp", ".txt", ".d"):
            add(path, "build/" + path.name)
    for path in sorted((work / "captures").iterdir()):
        assert path.suffix in (".json", ".bin")
        add(path, "captures/" + path.name)
    for name in ("v49-run-manifest.json", "v49-runs.jsonl.gz", "v49-summary.json",
                 "v49-offline-manifest.json", "v49-replay-runs.jsonl.gz", "v49-hsd-analysis.json",
                 "v49-hsd-analysis-process.json"):
        add(DATA / name, "results/" + name)
    for path in sorted((ROOT / ".tools").glob("v49-*-attempt*.log")):
        add(path, "logs/" + path.name)
    members = {name: {"source": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
               for name, path in files.items()}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, path in files.items():
            target.write(path, name)
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == set(members) and len(source.namelist()) == len(members)
        for name, value in members.items():
            payload = source.read(name)
            assert len(payload) == value["bytes"] and hashlib.sha256(payload).hexdigest() == value["sha256"]
            assert sha(Path(value["source"])) == value["sha256"]
    result = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "archive": str(archive), "zip_sha256": sha(archive), "members": members,
              "frozen_source": {"path": str(DATA / "ipm-v48-source.zip"), "sha256": sha(DATA / "ipm-v48-source.zip")},
              "scope": "保存新增来源、命令、日志、捕获数据及原始报告；不包含可执行文件、对象或第三方库。二进制身份由构建清单记录。"}
    with manifest.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({"members": len(members), "zip_bytes": archive.stat().st_size, "zip_sha256": result["zip_sha256"]}))


if __name__ == "__main__":
    main()
