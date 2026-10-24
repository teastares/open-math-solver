"""只读核验第47轮崩溃复现和调试记录，不启动求解器或调试器。"""

import argparse
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


DATA = Path(__file__).resolve().parent
ARTIFACTS = {}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ARTIFACTS[str(path)] = digest
    return digest


def parse(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "重复 JSON 键")
            result[key] = value
        return result
    def invalid(value):
        raise AssertionError("非有限 JSON 常量：" + value)
    def decimal(value):
        result = float(value)
        require(math.isfinite(result), "非有限 JSON 小数")
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=decimal)


def read(name):
    path = DATA / name
    sha(path)
    return parse(path.read_text(encoding="utf-8"))


def log_text(name):
    path = DATA / name
    sha(path)
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", path.read_text(encoding="utf-8", errors="replace"))


def verify():
    manifest = read("v50-crash-repro-manifest.json")
    require(manifest["diagnostic_only"] is True and manifest["repeats"] == 10 and
            manifest["solver_runs"] == 20 and manifest["wall_timeout"] == 35, "复现预算")
    require(sha(DATA / "v50-crash-repro-attempt1.py") == manifest["script_sha256"], "保留的首次脚本身份")
    require(sha(DATA / "v50-probe-runs.jsonl.gz") == manifest["origin_raw_sha256"], "原失败来源")
    origin_rows = [parse(line) for line in gzip.decompress((DATA / "v50-probe-runs.jsonl.gz").read_bytes()).splitlines()]
    origin, = [row for row in origin_rows if row["run_index"] == manifest["origin_run_index"]]
    require(origin["returncode"] == 3221225477 and origin["report"] is None and
            origin["classification"] == "invalid_json", "原始失败不能由后来正常运行替代")
    expected_hashes = {}
    for version in (48, 50):
        key = str(version)
        command = manifest["commands"][key]
        identity = read(f"ipm-v{version}-identity.json")
        require(command[1:] == origin["command"][1:], "两冻结版本使用同一原始命令")
        require(sha(command[0]) == manifest["solver_sha256"][key] == identity["solver_sha256"], "冻结程序身份")
        require(sha(command[2]) == manifest["input_sha256"] == origin["frozen_model_sha256"], "输入身份")
        expected_hashes[version] = {"solver": identity["solver_sha256"], "input": manifest["input_sha256"]}
    raw = DATA / "v50-crash-repro-runs.jsonl.gz"
    raw_sha = sha(raw)
    rows = [parse(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    order = [(version, repeat) for repeat in range(10) for version in ((48, 50) if repeat % 2 == 0 else (50, 48))]
    require([(row["version"], row["repetition"]) for row in rows] == order, "20次交替运行完整且唯一")
    counts = {str(version): {"reports": 0, "access_violations": 0, "other_process_failures": 0} for version in (48, 50)}
    for row in rows:
        version = row["version"]
        require(row["before"] == row["after"] == expected_hashes[version], "每次执行前后程序与输入身份")
        require(type(row["wall_seconds"]) in (int, float) and math.isfinite(row["wall_seconds"]) and
                row["wall_seconds"] >= 0 and row["error"] is None and row["stderr"] == "", "实际进程记录")
        if row["returncode"] == 3221225477:
            require(row["report"] is None and row["stdout"] == "", "访问违规无伪造报告")
            counts[str(version)]["access_violations"] += 1
        else:
            report = row["report"]
            require(isinstance(report, dict) and parse(row["stdout"]) == report, "合法报告与stdout一致")
            require(row["returncode"] == 4 and report["status"] == "numerical_error" and
                    report["backend"] == "ipm" and report["backend_version"] == f"ipm-cpu-0.{41 if version == 48 else 42}.0" and
                    report["input"] == origin["input"] and report["has_solution"] is False and
                    report["options"] == origin_rows[0]["report"]["options"], "退出4是合法数值失败，不能误计其他进程故障")
            counts[str(version)]["reports"] += 1
    require(counts == {"48": {"reports": 7, "access_violations": 3, "other_process_failures": 0},
                       "50": {"reports": 9, "access_violations": 1, "other_process_failures": 0}}, "复现实际分类")
    reviewed = read("v50-crash-repro-reviewed.json")
    original = read("v50-crash-repro-summary.json")
    require(reviewed["runs"] == original["runs"] == 20 and reviewed["raw_sha256"] == original["raw_sha256"] == raw_sha and
            reviewed["by_version"] == counts and reviewed["review_script_sha256"] == sha(DATA / "v50-crash-repro.py"), "修订摘要身份和分类")
    for key, actual in counts.items():
        require(original["by_version"][key] == dict(actual, other_process_failures=actual["reports"]), "原摘要分类错误已保留")

    target = manifest["commands"]["48"]
    debug_records = []
    for prefix, attempts in (("v50-lldb", [2]), ("v50-lldb-stack", [2, 3])):
        debug = read(prefix + "-manifest.json")
        result = read(prefix + "-result.json")
        require(debug["diagnostic_only"] is True and result["diagnostic_only"] is True and
                sha(DATA / (prefix + ".py")) == debug["script_sha256"] and
                sha(DATA / (prefix + "-manifest.json")) == result["manifest_sha256"], "调试脚本与清单身份")
        require(debug["target"] == target and debug["solver_sha256"] == expected_hashes[48]["solver"] and
                debug["input_sha256"] == manifest["input_sha256"] and
                sha(debug["debugger"]) == debug["debugger_sha256"] and
                debug["maximum_attempts"] == 10 and debug["wall_timeout"] == 60, "调试目标、工具和预算")
        require([record["attempt"] for record in result["records"]] == attempts, "实际调试次数")
        for record in result["records"]:
            base = prefix + f'-attempt{record["attempt"]}'
            paths = {str(DATA / (base + suffix)) for suffix in (".log", "-target.stdout", "-target.stderr", "-command.json")}
            require(set(record["files"]) == paths and all(sha(path) == digest for path, digest in record["files"].items()), "每份调试原件身份")
            command = read(base + "-command.json")
            stdout, stderr = DATA / (base + "-target.stdout"), DATA / (base + "-target.stderr")
            launch = f'process launch --stdout "{stdout.as_posix()}" --stderr "{stderr.as_posix()}"'
            option = "-k" if prefix.endswith("-stack") else "-o"
            expected = [debug["debugger"], "--no-lldbinit", "--batch", "-o", launch,
                        option, "thread backtrace all", option, "register read"]
            if prefix.endswith("-stack"):
                expected.extend(["-k", "memory read -fx -s8 -c20 $rsp"])
            expected.extend(["--", *target])
            require(command == expected, "实际调试命令与故障回调政策")
            text = log_text(base + ".log")
            caught = "0xc0000005" in text.lower()
            require(record["returncode"] == 0 and record["error"] is None and record["access_violation_caught"] == caught, "调试结果分类")
            require(stderr.read_bytes() == b"", "目标标准错误为空")
            if caught:
                require("dgemm_kernel_COOPERLAKE + 18685" in text and "<+18685>: retq" in text and
                        "Access violation reading location 0xffffffffffffffff" in text and stdout.read_bytes() == b"", "访问违规位置")
            else:
                require("exited with status = 4 (0x00000004)" in text and read(base + "-target.stdout")["status"] == "numerical_error", "正常数值失败退出")
            if prefix.endswith("-stack") and caught:
                require("(lldb) register read" in text and "(lldb) memory read -fx -s8 -c20 $rsp" in text and
                        "0x606575a378: 0xbefeb5f84f36294a" in text, "实际寄存器和栈内存已经捕获")
            debug_records.append({"base": base, "access_violation_caught": caught})
    manual = read("v50-lldb-attempt1-command.json")
    require(manual == [debug["debugger"], "--no-lldbinit", "--batch", "-o", "run", "-o", "thread backtrace all", "--", *target], "首次手动调试命令")
    manual_text = log_text("v50-lldb-attempt1.log")
    require("exited with status = 4 (0x00000004)" in manual_text and "0xc0000005" not in manual_text.lower() and
            "Command requires a process which is currently stopped" in manual_text, "首次调试是数值失败退出后无法取栈")
    debug_records.insert(0, {"base": "v50-lldb-attempt1", "access_violation_caught": False})
    initial, final = read("v50-verification-initial.json"), read("v50-verification.json")
    require(initial["passed"] is True and final["passed"] is True and initial["run_count"] == 138 and final["run_count"] == 502,
            "初检与完整核验通过且范围不同")
    for key in ("plan_sha256", "identity_sha256", "solver_sha256", "source_files", "linked_libraries", "old_trajectories", "gate", "self_test"):
        require(initial[key] == final[key], "初检与完整核验共同证据一致：" + key)
    for stage in ("development", "probe", "secondary-probe"):
        require(initial["stages"][stage] == final["stages"][stage], "初检阶段未被后来运行覆盖")
    sha(__file__)
    return {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "只读来源、原始分类及调试记录核验；未运行 LP、构建或调试器",
            "reproduction_runs": len(rows), "by_version": counts, "debugger_runs": len(debug_records),
            "debugger_access_violations": sum(record["access_violation_caught"] for record in debug_records),
            "debugger_records": debug_records, "initial_and_final_shared_evidence_equal": True,
            "artifacts": ARTIFACTS,
            "limits": ["首次手动调试没有运行时来源清单；这里只核命令、日志和当前冻结工具身份，不能补造当时的前后校验。",
                       "本核验确认崩溃早于新恢复策略，且两次调试停在 COOPERLAKE 的 ret 指令；未捕获入口 K 和写越界现场，不能据此单独证明缓冲容量为最终根因。",
                       "每个版本十次复现不构成稳定率估计；诊断数据不进入性能评分，原有失败保持原样。"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "reproduction_runs", "by_version", "debugger_runs", "debugger_access_violations")}, ensure_ascii=False))
