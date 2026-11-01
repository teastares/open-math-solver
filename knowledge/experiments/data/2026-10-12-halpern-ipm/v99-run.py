"""第96轮：冻结v98算法的216次阶段成本诊断，不执行性能门槛或HiGHS。"""
import argparse
import contextlib
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v99-profile"
PLAN_SHA = "523115acd8348c1d21c7f3bc9ce302fedcd2a5b630ec65d96ee1235c8226de15"
PARENT_RUN_SHA = "e0f879fa8f57bc49b4f7a09508828cadbf3411cab3eca5759f143675e0f9a41f"
ENVIRONMENT = "SOLVER_V99_PROFILE"
ALGORITHM_ENVIRONMENT = "SOLVER_V98_PRESOLVE"
PROFILE_MARKER = "V99_PROFILE "
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "profile")
PROFILE_STAGES = ("total", "canonicalize", "kkt_setup", "normal_setup", "symbolic_analysis", "factor",
    "normal_factor", "rectangular_assembly", "explicit_gram_assembly", "cholmod_numeric", "factor_check",
    "hsd_constant_direct", "hsd_direction_direct", "direct", "normal_solve", "cholmod_solve", "preconditioner",
    "hsd_column_product", "hsd_true_product", "coupled", "point_update", "assess_homogeneous", "verify_local",
    "initialize", "run_homogeneous", "corrector_rhs", "homogeneous_step", "cone_multiply", "cone_transpose_multiply",
    "cholmod_factor_adapter", "solve_factored")
PROFILE_FIELDS = ("schema", "stages", "clock_reads", "exclusive_sum_ns", "total_inclusive_ns", "conserved", "stack_empty", "counts_closed")
PROFILE_STAGE_FIELDS = ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")
MAPS = ("protected_before_and_after", "generated_inputs_before_and_after", "compile_inputs_after_and_final",
    "link_explicit_before_and_after", "test_link_explicit_before_and_after", "outputs")
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(DATA / "v98-run.py") == PARENT_RUN_SHA
_spec = importlib.util.spec_from_file_location("v99_readonly_v98_protocol", DATA / "v98-run.py")
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
base, benchmark, compare_lp = parent.base, parent.benchmark, parent.compare_lp
strict_json, read, save = parent.strict_json, parent.read, parent.save


def same(a, b):
    return base.typed_equal(a, b)


def save_once(path, value):
    if Path(path).exists():
        assert same(read(path), value), "禁止覆盖不一致证据：" + str(path)
    else:
        save(path, value)


def fixed_plan():
    assert sha(DATA / "v99-plan.json") == PLAN_SHA
    plan = read(DATA / "v99-plan.json")
    assert plan["diagnostic_only"] is True and plan["algorithm_parent_version"] == 98
    assert plan["maximum_solver_runs"] == 216 and plan["scheduled_runs"] == dict(zip(STAGES, (22, 12, 182)))
    assert plan["algorithm_environment"] == {ALGORITHM_ENVIRONMENT: "1"} and plan["profile_environment"] == ENVIRONMENT
    assert plan["profile_marker"] == "V99_PROFILE" and plan["inherited_markers"] == ["V98_PRESOLVE", "V98_COUPLED", "V98_ENERGY"]
    assert tuple(plan["stages"]) == PROFILE_STAGES and tuple(plan["profile_top_fields"]) == PROFILE_FIELDS
    assert tuple(plan["profile_stage_fields"]) == PROFILE_STAGE_FIELDS and type(plan["schema"]) is int and plan["schema"] == 1
    assert (plan["numerical_time_limit_seconds"], plan["process_timeout_seconds"], plan["thread_count"], plan["seed"]) == (5, 35, 1, 0)
    for key, hash_key in (("parent_closeout_file", "parent_closeout_sha256"), ("parent_plan_file", "parent_plan_sha256"),
        ("parent_identity_file", "algorithm_parent_identity_sha256"), ("baseline_verification_file", "baseline_verification_sha256"),
        ("parent_profile_helper", "parent_profile_helper_sha256")):
        assert sha(DATA / plan[key]) == plan[hash_key]
    assert sha(plan["parent_build_manifest"]) == plan["parent_build_manifest_sha256"]
    assert sha(plan["parent_reference"]) == plan["parent_reference_sha256"]
    for path, digest in plan["protected_parent_inputs"].items():
        assert sha(path) == digest, path
    for name in ("development", "probe", "netlib"):
        spec = plan["sets"][name]
        assert sha(spec["models_file"]) == spec["models_file_sha256"]
        assert len(spec["models"]) == {"development": 11, "probe": 6, "netlib": 91}[name]
        assert [item["model_index"] for item in spec["models"]] == list(range(len(spec["models"])))
        assert all(item["hash_error"] is None for item in spec["models"])
    return plan


def numeric_plan(plan):
    return dict(plan["solver_options"], compatibility_excluded_report_fields=["timings"])


def specification(stage, plan):
    assert stage in STAGES
    key = "netlib" if stage == "profile" else stage.removeprefix("compatibility-")
    return dict(plan["sets"][key], repeats=1)


def variants(stage):
    assert stage in STAGES
    return ("v98_reference", "profile_off") if stage in COMPAT else ("profile_off", "profile_on")


def clean_environment(prior, enabled):
    result = {key: value for key, value in prior.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    result.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    result.update({ALGORITHM_ENVIRONMENT: "1", ENVIRONMENT: "1" if enabled else "0"})
    return result


@contextlib.contextmanager
def solver_environment(enabled):
    old = dict(os.environ)
    try:
        os.environ.clear(); os.environ.update(clean_environment(old, enabled))
        yield
    finally:
        os.environ.clear(); os.environ.update(old)


def nonnegative(value):
    return type(value) is int and value >= 0


def valid_profile(value):
    if type(value) is not dict or set(value) != set(PROFILE_FIELDS) or type(value["schema"]) is not int or value["schema"] != 1:
        return False
    stages = value["stages"]
    if type(stages) is not dict or set(stages) != set(PROFILE_STAGES):
        return False
    for item in stages.values():
        if type(item) is not dict or set(item) != set(PROFILE_STAGE_FIELDS) or not all(nonnegative(v) for v in item.values()):
            return False
        if item["started"] != item["completed"] + item["unwound"] or item["exclusive_ns"] > item["inclusive_ns"]:
            return False
    return all(value[key] is True for key in ("conserved", "stack_empty", "counts_closed")) and \
        all(nonnegative(value[key]) for key in ("clock_reads", "exclusive_sum_ns", "total_inclusive_ns")) and \
        stages["total"]["started"] == 1 and \
        value["clock_reads"] == 2 * sum(item["started"] for item in stages.values()) and \
        value["exclusive_sum_ns"] == sum(item["exclusive_ns"] for item in stages.values()) == \
        value["total_inclusive_ns"] == stages["total"]["inclusive_ns"]


def inspect_markers(stderr, enabled):
    lines = stderr.splitlines()
    profile_lines = [line for line in lines if line.startswith(PROFILE_MARKER)]
    # 原三标记始终按启用的v98算法解释，不使用父协议的frozen_v66分支。
    inherited = parent.inspect_markers("\n".join(line for line in lines if not line.startswith(PROFILE_MARKER)), "diagnostic_on")
    profile, error = None, None
    if not enabled:
        if profile_lines:
            error = "unexpected_profile_marker"
    elif len(profile_lines) != 1:
        error = "profile_marker_count"
    else:
        try:
            profile = strict_json(profile_lines[0][len(PROFILE_MARKER):])
            if not valid_profile(profile):
                error = "invalid_profile_schema"
        except (TypeError, ValueError):
            error = "invalid_profile_json"
    return (*inherited, profile, error)


def configuration_matches(report, stage, variant, model, plan):
    return type(report) is dict and report.get("input") == model["input"] and report.get("backend") == "ipm" and \
        report.get("backend_version") == plan["solver_options"]["candidate_backend_version"] and \
        same(report.get("options"), plan["expected_report_options"])


def valid_process(row):
    return row.get("profile_marker_error") is None and type(row.get("profile_enabled")) is bool and \
        ((valid_profile(row.get("profile_marker"))) if row["profile_enabled"] else row.get("profile_marker") is None) and parent.valid_process(row)


def quality_fields(row, model, expected_solver, plan):
    value = parent.quality_fields(row, model, expected_solver, numeric_plan(plan))
    if not valid_process(dict(row, numerical_seconds=value["numerical_seconds"])):
        value.update(quality_failure="invalid_process", quality_pass=False,
            quality_penalized_wall_seconds=plan["process_timeout_seconds"],
            quality_penalized_numerical_seconds=plan["numerical_time_limit_seconds"])
    return value


def public_semantics(report):
    answer = copy.deepcopy(report)
    if type(answer) is dict:
        answer.pop("timings", None)
    return answer


def primary_semantics(value):
    answer = copy.deepcopy(value)
    if type(answer) is dict:
        answer.pop("timings", None)
        oracle = answer.get("oracle")
        if type(oracle) is dict:
            oracle.pop("total_seconds", None)
            for event in oracle.get("events", []):
                if type(event) is dict:
                    event.pop("timings", None)
    return answer


def pair_result(a, b, plan):
    public_equal = same(public_semantics(a.get("report")), public_semantics(b.get("report"))) and \
        same(a.get("returncode"), b.get("returncode")) and same(a.get("error"), b.get("error"))
    marker_equal = same(primary_semantics(a.get("presolve_marker")), primary_semantics(b.get("presolve_marker"))) and \
        all(same(a.get(key), b.get(key)) for key in ("coupled_marker", "energy_marker"))
    quality_equal = same(a.get("quality_pass"), b.get("quality_pass")) and same(a.get("quality_failure"), b.get("quality_failure"))
    valid = valid_process(a) and valid_process(b)
    budget = base.limited(a, numeric_plan(plan)) or base.limited(b, numeric_plan(plan))
    equal = public_equal and marker_equal and quality_equal
    category = "invalid_process" if not valid else "equal" if equal else "budget_limited" if budget else "different"
    return dict(category=category, equal=equal, valid_processes=valid, budget_limited=budget,
        public_report_equal_excluding_timings=public_equal, inherited_markers_equal_excluding_fixed_clocks=marker_equal,
        quality_equal=quality_equal, same_path_attribution=valid and equal and not budget,
        reference_quality_pass=a.get("quality_pass"), candidate_quality_pass=b.get("quality_pass"))


def freeze(work, plan):
    work = work.resolve()
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["diagnostic_only"] is True and build["production_changes"] is False
    assert build["public_lp_runs"] == 0 and build["algorithm_parent_version"] == 98
    assert set(build["math_self_tests"]) == {"ipm", "coupled"} and set(build["profile_self_tests"]) == {"off", "on"}
    legacy = read(DATA / "ipm-v66-identity.json")
    assert sha(DATA / "ipm-v66-identity.json") == plan["parent_identity_sha256"]
    assert build["current_files"] == build["frozen_files"] == legacy["files"] and build["actual_libraries"] == legacy["libraries"]
    assert len(legacy["files"]) == 95 and len(legacy["libraries"]) == 17
    assert sha(build["executable"]) == build["executable_sha256"] and Path(build["executable"]) == work / "solver-profile.exe"
    protected = {}
    for name in MAPS:
        for path, digest in build[name].items():
            assert sha(path) == digest, path
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for path, digest in plan["protected_parent_inputs"].items():
        assert protected[path] == digest
    for path, digest in read(DATA / plan["parent_identity_file"])["protected"].items():
        assert sha(path) == digest, path
        protected[path] = digest
    for name in plan["build_snapshot_files"]:
        path = str(DATA / name)
        assert path in protected
    for key in ("files", "libraries"):
        for relative, digest in legacy[key].items():
            path = str(ROOT / relative)
            assert sha(path) == digest
            protected[path] = digest
    protected.update({str(build_path): sha(build_path), str(DATA / "v99-run.py"): sha(__file__)})
    identity = dict(plan_sha256=PLAN_SHA, diagnostic_only=True, algorithm_parent_version=98,
        build_manifest=str(build_path), build_manifest_sha256=sha(build_path),
        candidate=build["executable"], candidate_sha256=build["executable_sha256"],
        reference=plan["parent_reference"], reference_sha256=plan["parent_reference_sha256"], protected=protected,
        math_self_tests=build["math_self_tests"], profile_self_tests=build["profile_self_tests"])
    save_once(DATA / "v99-run-identity.json", identity)
    return identity


def stage_summary(stage, rows, plan):
    expected = specification(stage, plan)
    pairs = []
    for model in expected["models"]:
        chosen = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
        assert set(chosen) == set(variants(stage))
        pairs.append(dict(model_index=model["model_index"], input=model["input"],
            **pair_result(*(chosen[name] for name in variants(stage)), plan)))
    return dict(plan_sha256=PLAN_SHA, diagnostic_only=True, stage=stage, run_count=len(rows),
        variants={name: dict(run_count=sum(row["requested_variant"] == name for row in rows),
            quality_pass_count=sum(row["requested_variant"] == name and row["quality_pass"] for row in rows),
            valid_process_count=sum(row["requested_variant"] == name and valid_process(row) for row in rows)) for name in variants(stage)},
        valid_process_count=sum(valid_process(row) for row in rows),
        profile_marker_count=sum(row["profile_marker"] is not None for row in rows),
        valid_profile_marker_count=sum(valid_profile(row["profile_marker"]) for row in rows),
        profile_marker_error_count=sum(row["profile_marker_error"] is not None for row in rows),
        pairs=pairs, equal_pairs=sum(pair["equal"] for pair in pairs),
        same_path_pairs=sum(pair["same_path_attribution"] for pair in pairs),
        budget_pairs=sum(pair["budget_limited"] for pair in pairs),
        different_pairs=sum(pair["category"] == "different" for pair in pairs),
        invalid_pairs=sum(pair["category"] == "invalid_process" for pair in pairs),
        performance_gate_applied=False, highs_runs=0)


def archive(stage):
    plan = fixed_plan()
    identity = read(DATA / "v99-run-identity.json")
    manifest = read(DATA / f"v99-{stage}-manifest.json")
    raw_path = DATA / f"v99-{stage}-runs.jsonl.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        rows = [strict_json(line) for line in stream]
    spec = specification(stage, plan)
    assert manifest["plan_sha256"] == PLAN_SHA and manifest["repeats"] == 1
    assert manifest["identity_sha256"] == sha(DATA / "v99-run-identity.json")
    assert same(manifest["models"], spec["models"]) and manifest["variants"] == list(variants(stage))
    assert len(rows) == plan["scheduled_runs"][stage]
    for index, ((model, repeat, position, variant), row) in enumerate(zip(compare_lp.schedule(spec["models"], 1, variants(stage)), rows)):
        assert row["run_index"] == index and row["model_index"] == model["model_index"] and row["repetition"] == repeat
        assert row["position_in_pair"] == position and row["requested_variant"] == variant
        executable = identity["reference"] if variant == "v98_reference" else identity["candidate"]
        expected_sha = identity["reference_sha256"] if variant == "v98_reference" else identity["candidate_sha256"]
        assert row["command"][:3] == [executable, "solve", model["input"]]
        assert parent.valid_command(row, "diagnostic_on", numeric_plan(plan))
        assert row["solver_sha256_before"] == row["solver_sha256_after"] == row["frozen_solver_sha256"] == expected_sha == sha(executable)
        assert row["model_sha256"] == row["frozen_model_sha256"] == row["model_sha256_after"] == model["sha256"] == sha(model["input"])
        enabled = variant == "profile_on"
        assert row["profile_enabled"] is enabled and row["profile_environment"] == {ENVIRONMENT: "1" if enabled else "0"}
        assert row["presolve_environment"] == {ALGORITHM_ENVIRONMENT: "1"}
        parsed = inspect_markers(row["stderr"], enabled)
        keys = ("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error",
            "energy_marker", "energy_marker_error", "profile_marker", "profile_marker_error")
        assert all(same(row[key], value) for key, value in zip(keys, parsed))
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        assert all(same(row[key], value) for key, value in quality_fields(row, model, expected_sha, plan).items())
        if type(row["report"]) is dict:
            assert same(strict_json(row["stdout"]), row["report"])
    summary = stage_summary(stage, rows, plan)
    summary["raw_sha256"] = sha(raw_path)
    assert same(read(DATA / f"v99-{stage}-summary.json"), summary)
    return manifest, summary, rows


def compatibility():
    pairs = []
    for stage in COMPAT:
        _, summary, _ = archive(stage)
        pairs.extend(dict(stage=stage, **pair) for pair in summary["pairs"])
    assert len(pairs) == 17
    return dict(passed=all(pair["valid_processes"] and pair["public_report_equal_excluding_timings"] for pair in pairs),
        diagnostic_only=True, plan_sha256=PLAN_SHA, runs=34, pairs=pairs,
        complete_public_report_equal_pairs=sum(pair["public_report_equal_excluding_timings"] for pair in pairs),
        inherited_semantic_equal_pairs=sum(pair["inherited_markers_equal_excluding_fixed_clocks"] for pair in pairs))


def run(stage, work):
    plan = fixed_plan()
    identity = freeze(work, plan)
    spec = specification(stage, plan)
    for model in spec["models"]:
        assert sha(model["input"]) == model["sha256"]
    manifest = dict(schema=1, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA, diagnostic_only=True, stage=stage, identity_sha256=sha(DATA / "v99-run-identity.json"),
        models_file=spec["models_file"], models_file_sha256=spec["models_file_sha256"], models=spec["models"], repeats=1,
        variants=list(variants(stage)), reference_variant=variants(stage)[0], candidate_variant=variants(stage)[1],
        time_limit=5, wall_timeout=35, threads=1, seed=0,
        execution="串行、固定索引奇偶轮转；所有失败保留；216次诊断，无性能门槛、无HiGHS、无重跑替换。")
    save(DATA / f"v99-{stage}-manifest.json", manifest)
    rows = []
    raw_path = DATA / f"v99-{stage}-runs.jsonl.gz"
    with gzip.open(raw_path, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in compare_lp.schedule(spec["models"], 1, variants(stage)):
            executable = Path(identity["reference"] if variant == "v98_reference" else identity["candidate"])
            expected_sha = identity["reference_sha256"] if variant == "v98_reference" else identity["candidate_sha256"]
            before = sha(executable)
            enabled = variant == "profile_on"
            with solver_environment(enabled):
                row = benchmark.run_one(parent.options_for(stage, "diagnostic_on", executable, numeric_plan(plan)),
                    Path(model["input"]), 0, len(rows))
            parsed = inspect_markers(row["stderr"], enabled)
            row.update(dict(zip(("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error",
                "energy_marker", "energy_marker_error", "profile_marker", "profile_marker_error"), parsed)))
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position,
                requested_variant=variant, frozen_model_sha256=model["sha256"], model_sha256_after=sha(model["input"]),
                solver_sha256_before=before, solver_sha256_after=sha(executable), frozen_solver_sha256=expected_sha,
                presolve_environment={ALGORITHM_ENVIRONMENT: "1"}, profile_enabled=enabled,
                profile_environment={ENVIRONMENT: "1" if enabled else "0"},
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_sha, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
            rows.append(row)
            print(json.dumps(dict(stage=stage, index=row["run_index"], model=Path(model["input"]).stem, variant=variant,
                status=row["status"], quality_pass=row["quality_pass"], profile_marker_error=row["profile_marker_error"]), ensure_ascii=False), flush=True)
    summary = stage_summary(stage, rows, plan)
    summary["raw_sha256"] = sha(raw_path)
    save(DATA / f"v99-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps(dict(stage=stage, diagnostic_only=True, run_count=len(rows), variants=summary["variants"]), ensure_ascii=False), flush=True)


def self_check():
    plan = fixed_plan()
    blank = dict.fromkeys(PROFILE_STAGE_FIELDS, 0)
    stages = {key: dict(blank) for key in PROFILE_STAGES}
    stages["total"] = dict(inclusive_ns=12, exclusive_ns=5, started=1, completed=1, unwound=0)
    stages["factor"] = dict(inclusive_ns=7, exclusive_ns=7, started=2, completed=1, unwound=1)
    good = dict(schema=1, stages=stages, clock_reads=6, exclusive_sum_ns=12, total_inclusive_ns=12,
        conserved=True, stack_empty=True, counts_closed=True)
    assert valid_profile(good)
    mutations = [("schema", True), ("clock_reads", 5), ("exclusive_sum_ns", 11), ("conserved", 1), ("stack_empty", False)]
    for key, value in mutations:
        bad = copy.deepcopy(good); bad[key] = value
        assert not valid_profile(bad)
    for key, value in (("started", True), ("completed", 0), ("exclusive_ns", 8)):
        bad = copy.deepcopy(good); bad["stages"]["factor"][key] = value
        assert not valid_profile(bad)
    bad = copy.deepcopy(good); bad["stages"]["extra"] = dict(blank)
    assert not valid_profile(bad)
    inherited = "\n".join(prefix + "{}" for prefix in ("V98_PRESOLVE ", "V98_COUPLED ", "V98_ENERGY "))
    text = inherited + "\n" + PROFILE_MARKER + json.dumps(good)
    assert inspect_markers(text, True)[-2:] == (good, None)
    assert inspect_markers(text, False)[-1] == "unexpected_profile_marker"
    assert inspect_markers(text + "\n" + PROFILE_MARKER + json.dumps(good), True)[-1] == "profile_marker_count"
    assert inspect_markers(inherited, False)[-2:] == (None, None)
    prior = {"solver_v1": "1", "SoLvEr_V98_PrEsOlVe": "0", "SOLVER_V99_PROFILE": "9", "omp_num_threads": "9", "PATH": "保留"}
    for enabled in (False, True):
        value = clean_environment(prior, enabled)
        assert value == {"PATH": "保留", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            ALGORITHM_ENVIRONMENT: "1", ENVIRONMENT: "1" if enabled else "0"}
    old = dict(os.environ)
    with solver_environment(True):
        with solver_environment(False):
            assert os.environ[ALGORITHM_ENVIRONMENT] == "1" and os.environ[ENVIRONMENT] == "0"
        assert os.environ[ENVIRONMENT] == "1"
    assert dict(os.environ) == old
    assert sum(len(list(compare_lp.schedule(specification(stage, plan)["models"], 1, variants(stage)))) for stage in STAGES) == 216
    marker = dict(timings={"total": 1}, budget={"checks": 2, "loop_units": 3}, oracle=dict(total_seconds=1,
        events=[dict(timings={"total": 1}, decision="保留")]))
    changed = copy.deepcopy(marker); changed["oracle"]["events"][0]["timings"]["total"] = 2
    assert same(primary_semantics(marker), primary_semantics(changed))
    changed["budget"]["checks"] = 3
    assert not same(primary_semantics(marker), primary_semantics(changed))
    return dict(passed=True, public_lp_runs=0, profile_schema_cases=10, marker_cases=4, environment_cases=4,
        fixed_clock_mask_cases=2, algorithm_parent_version=98, scheduled_runs=plan["scheduled_runs"], maximum_solver_runs=216,
        performance_gate_applied=False, highs_runs=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage == "compatibility":
        result = compatibility(); save_once(DATA / "v99-compatibility.json", result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
