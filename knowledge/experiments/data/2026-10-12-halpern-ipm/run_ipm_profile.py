"""串行采集固定开发集阶段计时，并核对插桩前后的非时间报告。"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise RuntimeError("输出必须是工作区中尚不存在的目录")
    output.mkdir(parents=True)
    models_file = root / "benchmarks/development.txt"
    models = [(models_file.parent / line).resolve() for line in
              models_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    configs = [("qdldl", "on", "off"), ("normal", "on", "on"), ("factorhighs", "off", "off")]
    manifest = {"script_sha256": sha(Path(__file__)), "repeats": 3, "configs": configs,
                "profile": {"path": str(args.profile.resolve()), "sha256": sha(args.profile)},
                "reference": {"path": str(args.reference.resolve()), "sha256": sha(args.reference)},
                "models": [{"path": str(p), "sha256": sha(p)} for p in models],
                "models_file_sha256": sha(models_file), "time_limit": 5, "wall_timeout": 35,
                "scope": "插桩阶段诊断，绝非正式加速比测量"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(output / "runs.jsonl.gz", "wt", encoding="utf-8") as raw:
        for linear, singleton, fixed in configs:
            summary = {"linear": linear, "singleton": singleton, "fixed": fixed,
                       "runs": 0, "trajectory_equal": 0, "optimal": 0, "stages": {}}
            for model in models:
                common = ["solve", str(model), "--backend", "ipm", "--time-limit", "5",
                          "--threads", "1", "--seed", "0", "--ipm-linear-solver", linear,
                          "--ipm-singleton-elimination", singleton, "--ipm-kkt-equilibration", "on",
                          "--ipm-fixed-variable-elimination", fixed, "--ipm-linear-accuracy", "strict"]
                reference = None
                for repeat, executable in [(-1, args.reference)] + [(i, args.profile) for i in range(3)]:
                    command = [str(executable.resolve())] + common
                    process = subprocess.run(command, capture_output=True, timeout=35)
                    report = json.loads(process.stdout)
                    stderr = process.stderr.decode("utf-8", errors="replace")
                    row = {"linear": linear, "input": str(model), "repeat": repeat,
                           "command": command, "returncode": process.returncode, "report": report, "stderr": stderr}
                    raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                    comparable = {k: v for k, v in report.items() if k != "timings"}
                    if repeat == -1:
                        reference = comparable
                        continue
                    summary["runs"] += 1
                    summary["trajectory_equal"] += comparable == reference
                    summary["optimal"] += report["status"] == "optimal"
                    lines = [line for line in stderr.splitlines() if line.startswith("IPM_PROFILE ")]
                    if len(lines) != 1:
                        raise RuntimeError("阶段报告数量不为一")
                    stages = json.loads(lines[0][len("IPM_PROFILE "):])
                    for name, entry in stages.items():
                        target = summary["stages"].setdefault(name, {"inclusive": 0, "exclusive": 0, "calls": 0})
                        for metric, value in entry.items():
                            target[metric] += value
                    exclusive = sum(entry["exclusive"] for entry in stages.values())
                    if abs(exclusive - stages["total"]["inclusive"]) > 1e-9:
                        raise RuntimeError("包含与独占计时不守恒")
            summaries.append(summary)
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
