"""只桥接诊断输出候选身份，完整复用冻结120的数学校验和118运输层。"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import build_experimental as B

PARENT_FILE = ROOT / ".tools/v120-benchmark.py"
PARENT_SHA = "44cd0c9b1934c5843b116db5ffa75d1913682c89c6ffa16290db47b4b5e86c20"
B.require(B.sha(PARENT_FILE) == PARENT_SHA, "冻结120验证器源码变化")
spec = importlib.util.spec_from_file_location("v121_frozen_v120_protocol", PARENT_FILE)
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def contract(plan):
    value = plan["hybrid_protocol"]
    B.require(set(value) == {"validator_file", "validator_sha256", "environment", "marker", "schema",
        "on_backend_version", "off_backend_version", "marker_on_only"} and plan["candidate_version"] == 121 and
        plan["source_count"] == 15 and plan["new_algorithm_round"] is True and plan["production_changes"] is False and
        plan["candidate_backend_version"] == value["on_backend_version"] == Q.NEW_VERSION == "ipm-ipx-crossover-cpu-0.3.0" and
        value["off_backend_version"] == Q.P.OLD_VERSION and value["environment"] == Q.P.ENVIRONMENT and
        value["marker"] == Q.P.MARKER and type(value["schema"]) is int and value["schema"] == 3 and
        value["marker_on_only"] is True, "121只能改变诊断输出实现及实验身份")
    own = Path(__file__).resolve()
    B.require((ROOT / value["validator_file"]).resolve() == own and B.sha(own) == value["validator_sha256"], "121身份桥源码变化")
    tools = {str((ROOT / name).resolve()): digest for name, digest in plan["tool_sources"].items()}
    required = (own, PARENT_FILE, Q.PARENT_FILE, Path(B.__file__).resolve(), Path(Q.P.C.__file__).resolve(),
        ROOT / "scripts/record_experiment.py")
    B.require(all(str(path) in tools for path in required) and tools[str(own)] == value["validator_sha256"] and
        tools[str(PARENT_FILE)] == PARENT_SHA and tools[str(Q.PARENT_FILE)] == Q.PARENT_SHA,
        "计划必须锁定121/120/118及稳定构建、评测和调用记录六项来源")
    B.verify(tools)
    return value


def provenance(output, plan, create):
    for source, digest in ((Path(__file__).resolve(), plan["hybrid_protocol"]["validator_sha256"]),
        (PARENT_FILE, PARENT_SHA), (Q.PARENT_FILE, Q.PARENT_SHA)):
        target = output / "provenance" / source.name
        if create and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        B.require(target.is_file() and B.sha(target) == digest, "121及冻结验证器的逐字快照缺失或变化")


# 仅改变独立导入实例的身份入口；旧文件及数学/质量验证函数保持原样。
Q.P.contract, Q.P.provenance = contract, provenance

if __name__ == "__main__":
    Q.P.main()
