"""串行运行固定开发与诊断输入，保存全部日志并核对非时间报告。"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise RuntimeError("诊断输出必须是工作区中尚不存在的目录")
    output.mkdir(parents=True)
    models_file = root / "benchmarks/development.txt"
    models = [(models_file.parent / line).resolve() for line in models_file.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]
    models += [root / f".deps/netlib/{name}.mps" for name in ("dfl001", "finnis", "greenbea", "etamacro", "greenbeb")]
    configs = ["normal_augmented", "normal_multifrontal"]
    executables = {"reference": args.reference.resolve(), "probe": args.probe.resolve()}
    manifest = {"script_sha256": sha(Path(__file__)), "models_file_sha256": sha(models_file),
                "models": [{"path": str(p), "sha256": sha(p)} for p in models], "configs": configs,
                "executables": {k: {"path": str(p), "sha256": sha(p)} for k, p in executables.items()},
                "time_limit": 15, "wall_timeout": 90, "repeats": 1, "fixed_variables": "on",
                "scope": "只作误差来源诊断，额外乘积/复制/打印均影响计时，不是公开性能验收"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(output / "runs.jsonl.gz", "wt", encoding="utf-8") as raw:
        for linear in configs:
            for model in models:
                reports = {}
                observations = []
                direct_observations = []
                first_directions = []
                for variant, executable in executables.items():
                    command = [str(executable), "solve", str(model), "--backend", "ipm", "--time-limit", "15",
                               "--threads", "1", "--seed", "0", "--ipm-linear-solver", linear,
                               "--ipm-singleton-elimination", "on", "--ipm-kkt-equilibration", "on",
                               "--ipm-fixed-variable-elimination", "on", "--ipm-linear-accuracy", "strict"]
                    process = subprocess.run(command, capture_output=True, timeout=90)
                    report = json.loads(process.stdout)
                    stderr = process.stderr.decode("utf-8", errors="replace")
                    raw.write(json.dumps({"input": str(model), "linear": linear, "variant": variant,
                                          "command": command, "returncode": process.returncode,
                                          "report": report, "stderr": stderr}, ensure_ascii=False, allow_nan=False) + "\n")
                    raw.flush()
                    reports[variant] = {k: v for k, v in report.items() if k != "timings"}
                    if variant == "probe":
                        last_direct = None
                        for line in stderr.splitlines():
                            if line.startswith("DIRECT_REGULARIZED "):
                                last_direct = json.loads(line[len("DIRECT_REGULARIZED "):])
                                if last_direct.get("format_sentinel") != 1.25:
                                    raise RuntimeError("完整 KKT 诊断序列化自检失败")
                                direct_observations.append(last_direct)
                                continue
                            if line.startswith("改进 ") and last_direct:
                                parts = line.split()
                                if int(parts[3]) == 0:
                                    first_directions.append({"outer_iterations": int(parts[1]),
                                                             "trace_residual": float(parts[5]),
                                                             "trace_tolerance": float(parts[7]), **last_direct})
                            if not line.startswith("NORMAL_RESIDUAL "):
                                continue
                            row = json.loads(line[len("NORMAL_RESIDUAL "):])
                            if row.get("format_sentinel") != 1.25:
                                raise RuntimeError("诊断数值序列化哨兵不匹配，禁止解释这些误差量")
                            row["outside_double_range"] = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
                            for key in row["outside_double_range"]:
                                row[key] = None
                            observations.append(row)
                first = [row for row in observations if row["solve"] == 1]
                def quality(row):
                    denominator = 1e-12 + 1e-10 * max(row["kkt_primal_rhs"], row["kkt_dual_rhs"])
                    return max(row["kkt_primal_residual"], row["kkt_dual_residual"]) / denominator
                summary = {"model": model.stem, "linear": linear,
                           "trajectory_equal": reports["reference"] == reports["probe"],
                           "reference_status": reports["reference"]["status"], "probe_status": reports["probe"]["status"],
                           "reference_iterations": reports["reference"]["ipm_iterations"],
                           "probe_iterations": reports["probe"]["ipm_iterations"],
                           "observations": len(observations), "first_solves": first,
                           "direct_observations": len(direct_observations), "first_directions": first_directions,
                           "last": observations[-1] if observations else None,
                           "worst_factor_berr": max(observations, key=lambda x: x["factor_berr"] or 0) if observations else None,
                           "worst_kkt": max(observations, key=quality) if observations else None}
                summaries.append(summary)
                print(json.dumps({k: summary[k] for k in ("model", "linear", "trajectory_equal", "probe_status", "probe_iterations", "observations")}, ensure_ascii=False), flush=True)
    for key, executable in executables.items():
        if sha(executable) != manifest["executables"][key]["sha256"]:
            raise RuntimeError("诊断期间可执行文件发生变化")
    for frozen in manifest["models"]:
        if sha(Path(frozen["path"])) != frozen["sha256"]:
            raise RuntimeError("诊断期间模型发生变化")
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
