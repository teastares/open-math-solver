"""归档第50轮独立诊断的新增来源、命令、快照和分析，不重跑LP。"""
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
    archive, manifest = DATA / "v53-diagnostic-artifacts.zip", DATA / "v53-archive-manifest.json"
    assert not archive.exists() and not manifest.exists()
    run = json.loads((DATA / "v53-run-manifest.json").read_text(encoding="utf-8"))
    work = Path(run["work"])
    files = {}

    def add(path, name):
        assert name not in files and path.is_file()
        files[name] = path

    for pattern in ("v53-*.py", "v53-*.hpp", "v53-*.cpp", "v53-plan.json"):
        for path in sorted(DATA.glob(pattern)):
            add(path, "sources/" + path.name)
    for path in sorted(work.iterdir()):
        if path.is_file() and path.suffix in (".json", ".bin", ".cpp", ".txt", ".d"):
            add(path, "build/" + path.name)
    for path in sorted((work / "captures").iterdir()):
        assert path.suffix in (".json", ".bin")
        add(path, "captures/" + path.name)
    for name in ("v53-run-manifest.json", "v53-runs.jsonl.gz", "v53-summary.json", "v53-hsd-analysis.json",
                 "v53-hsd-analysis-process.json", "v53-runner-self-check.json", "v53-verification.json"):
        add(DATA / name, "results/" + name)
    for path in sorted(DATA.glob("v53-*-attempt*.json")):
        add(path, "attempts/" + path.name)
    members = {name: {"source": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for name, path in files.items()}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name, path in files.items():
            target.write(path, name)
    with zipfile.ZipFile(archive) as source:
        assert len(source.namelist()) == len(members) and set(source.namelist()) == set(members)
        for name, value in members.items():
            payload = source.read(name)
            assert len(payload) == value["bytes"] and hashlib.sha256(payload).hexdigest() == value["sha256"]
            assert sha(Path(value["source"])) == value["sha256"]
    result = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "archive": str(archive),
        "zip_sha256": sha(archive), "members": members,
        "frozen_source": {"path": str(DATA / "ipm-v52-source.zip"), "sha256": sha(DATA / "ipm-v52-source.zip")},
        "scope": "只保存新增来源、转换结果、命令、快照与证据；完整生产源码复用v52包，程序/对象/库仅记身份。"}
    with manifest.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({"members": len(members), "zip_bytes": archive.stat().st_size, "zip_sha256": result["zip_sha256"]}))


if __name__ == "__main__":
    main()
