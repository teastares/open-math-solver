"""第63轮CHOLMOD矩形正规方程：先验旧路径兼容，再按固定开发/探针门槛扩集。"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN = DATA / "v66-plan.json"
PLAN_SHA = "5f134f630f8f969097c2b9c2b85912a89b4f20ef224043303367d43754d1dd06"
EXE = ROOT / ".tools/solver-ipm-v66.exe"
PARENT = ROOT / ".tools/solver-ipm-v57.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
NEW_FIELDS = ()
ABSENT_FIELDS = ("ipm_extended_krylov_attempts", "ipm_extended_krylov_successes",
                 "ipm_homogeneous_scalar_checks", "ipm_homogeneous_scalar_rejections")
COUNT_FIELDS = ('ipm_iterations', 'ipm_factorizations', 'ipm_refinement_steps', 'ipm_weighted_krylov_steps', 'ipm_krylov_recycle_projections', 'ipm_krylov_recycle_acceptances', 'ipm_krylov_recycle_vectors', 'ipm_krylov_recycle_peak_dimension', 'ipm_regularization_searches', 'ipm_regularization_reductions', 'ipm_unshifted_factorizations', 'ipm_full_newton_preconditioner_solves', 'ipm_full_newton_products', 'ipm_bounded_newton_returns', 'ipm_bounded_newton_inexact_returns', 'ipm_homogeneous_constant_solves', 'ipm_homogeneous_constant_repairs', 'ipm_homogeneous_constant_repair_successes', 'ipm_homogeneous_direction_solves', 'ipm_homogeneous_scaled_products', 'ipm_homogeneous_scaled_refinement_steps', 'ipm_homogeneous_scaled_inexact_returns', 'ipm_direction_retries', 'ipm_direction_retry_successes', 'ipm_schur_residual_products', 'ipm_schur_refinement_steps', 'ipm_schur_refinement_acceptances', 'ipm_kkt_dimension', 'ipm_factor_nonzeros', 'ipm_structural_fallbacks', 'ipm_eliminated_singletons', 'ipm_regularized_pivots', 'ipm_backward_error_acceptances', 'ipm_inexact_acceptances', 'ipm_inexact_prefilter_rejections', 'ipm_eliminated_fixed_variables', 'ipm_presolved_singleton_rows', 'ipm_presolved_fixed_columns', 'ipm_central_correction_attempts', 'ipm_central_correction_acceptances')
OPTIONAL_COUNT_FIELDS = ("ipm_kkt_dimension", "ipm_factor_nonzeros", "ipm_eliminated_singletons")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def plan():
    assert sha(PLAN) == PLAN_SHA
    value = read(PLAN)
    assert sha(DATA / "ipm-v57-identity.json") == value["parent_identity_sha256"]
    assert sha(DATA / value["evidence_file"]) == value["evidence_sha256"]
    ancestor = value["implementation_ancestor"]
    assert sha(DATA / ancestor["identity_file"]) == ancestor["identity_sha256"]
    assert read(DATA / ancestor["identity_file"])["source_zip_sha256"] == ancestor["source_zip_sha256"]
    assert sha(DATA / value["starting_state"]["file"]) == value["starting_state"]["sha256"]
    assert (value["parent_version"], value["reference_version"], value["candidate_version"],
            value["reference_objective_scaling"], value["candidate_objective_scaling"]) == (57, 57, 66, "off", "off")
    assert value["candidate_backend_version"] == "ipm-cpu-0.53.0"
    assert value["new_report_fields"] == []
    assert value["compatibility_excluded_report_fields"] == ["timings", "backend_version"]
    assert (value["candidate_linear_solver"], value["reference_linear_solver"], value["compatibility_linear_solver"]) == ("normal_cholmod_aat", "normal_cholmod", "normal_cholmod")
    assert sha(DATA / value["baseline_check"]["file"]) == value["baseline_check"]["sha256"]
    identity = DATA / "ipm-v66-identity.json"
    if identity.exists():
        frozen = read(identity)
        assert frozen["runner_sha256"] == sha(__file__) and frozen["plan_sha256"] == PLAN_SHA
    return value


def environment():
    result = os.environ.copy()
    result.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    for key in tuple(result):
        if key.startswith("SOLVER_V"):
            result.pop(key)
    result.pop("OPENBLAS_CORETYPE", None)
    return result


def check_solver(version):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    assert sha(executable) == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    return executable


def specification(stage, frozen):
    assert stage in STAGES
    value = copy.deepcopy(frozen["sets"]["netlib" if stage in ("netlib", "highs") else stage.removeprefix("compatibility-")])
    if stage in COMPAT:
        value["repeats"] = 1
    return value


def accuracy(stage, side, frozen):
    return frozen["compatibility_accuracy"] if stage in COMPAT else frozen[side + "_accuracy"]


def linear_solver(stage, side, frozen):
    return frozen["compatibility_linear_solver"] if stage in COMPAT else frozen[side + "_linear_solver"]


def command_for(stage, frozen, output):
    expected = specification(stage, frozen)
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--reference-solver", str(EXE if stage == "highs" else PARENT),
               "--models-file", expected["models_file"], "--repeats", str(expected["repeats"]),
               "--time-limit", str(frozen["time_limit"]), "--wall-timeout", str(frozen["wall_timeout"]),
               "--threads", str(frozen["threads"]), "--seed", str(frozen["seed"]),
               "--candidate-backend", "ipm", "--reference-backend", "highs" if stage == "highs" else "ipm",
               "--output-dir", str(output)]
    for prefix, side in (("--", "candidate"), ("--reference-", "reference")):
        if stage == "highs" and side == "reference":
            for name, value in frozen["highs_reference"].items():
                command += [prefix + "highs-" + name, value]
            continue
        command += [prefix + "iteration-limit", str(frozen["max_iterations"]),
                    prefix + "scaling-passes", str(frozen["scaling_passes"]),
                    prefix + "ipm-method", frozen["method"],
                    prefix + "ipm-linear-accuracy", accuracy(stage, side, frozen),
                    prefix + "ipm-linear-solver", linear_solver(stage, side, frozen),
                    prefix + "ipm-direction-retries", "0"]
        for key, value in frozen["shared_ipm"].items():
            command += [prefix + "ipm-" + key, value]
    command += ["--ipm-objective-scaling", frozen["candidate_objective_scaling"]]
    if stage != "highs":
        command += ["--reference-ipm-objective-scaling", frozen["reference_objective_scaling"]]
    return command


def archive(stage):
    frozen = plan()
    expected = specification(stage, frozen)
    manifest = read(DATA / f"v66-{stage}-manifest.json")
    summary = read(DATA / f"v66-{stage}-summary.json")
    rows = [benchmark.strict_json(line) for line in gzip.decompress((DATA / f"v66-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
    assert manifest["models"] == expected["models"] and manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[key] == frozen[key]
    assert manifest["repeats"] == expected["repeats"]
    assert manifest["ipm_linear_accuracy"] == accuracy(stage, "candidate", frozen)
    assert manifest["ipm_linear_solver"] == linear_solver(stage, "candidate", frozen) and manifest["ipm_method"] == frozen["method"]
    if stage != "highs":
        assert manifest["reference_ipm_linear_accuracy"] == accuracy(stage, "reference", frozen)
        assert manifest["reference_ipm_linear_solver"] == linear_solver(stage, "reference", frozen)
        assert manifest["reference_ipm_method"] == frozen["method"]
    reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
    assert manifest["ipm_objective_scaling"] == frozen["candidate_objective_scaling"]
    assert manifest["reference_ipm_objective_scaling"] == (None if stage == "highs" else frozen["reference_objective_scaling"])
    for variant, version in ((reference, 66 if stage == "highs" else 57), (candidate, 66)):
        solver = manifest["solvers"][variant]
        assert Path(solver["path"]) == check_solver(version)
        assert solver["sha256"] == sha(solver["path"]) and solver["hash_error"] is None
    validate_rows(stage, frozen, manifest, rows)
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v66-{stage}"))
    assert recomputed == summary
    return manifest, summary, rows


def typed_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    return a == b


def expected_options(stage, side, frozen):
    value = copy.deepcopy(frozen["expected_report_options"])
    if stage == "highs" and side == "reference":
        value.update(backend="highs", ipm=None, highs={"solver": "choose", "run_crossover": "on", "presolve": "choose"})
    else:
        value["ipm"]["linear_accuracy"] = accuracy(stage, side, frozen)
        value["ipm"]["linear_solver"] = linear_solver(stage, side, frozen)
    return value


def valid_counts(report, version, backend, mode):
    if not isinstance(report, dict) or any(field in report for field in ABSENT_FIELDS):
        return False
    if not all(field in report for field in COUNT_FIELDS):
        return False
    if backend != "ipm":
        return all(report[field] is None for field in COUNT_FIELDS)
    for field in COUNT_FIELDS:
        value = report[field]
        if value is None and field in OPTIONAL_COUNT_FIELDS:
            continue
        if type(value) is not int or not 0 <= value <= 9223372036854775807:
            return False
    weighted, refinements = (report[field] for field in ("ipm_weighted_krylov_steps", "ipm_refinement_steps"))
    repairs, successes, constants = (report["ipm_homogeneous_constant_" + suffix]
                                    for suffix in ("repairs", "repair_successes", "solves"))
    return successes <= repairs <= constants and weighted <= refinements and (mode != "homogeneous_repair" or weighted == 0)


def valid_factor_path(report, backend):
    if "ipm_factor_backend" not in report:
        return False
    actual = report["ipm_factor_backend"]
    if backend != "ipm":
        return actual is None
    requested = report.get("options", {}).get("ipm", {}).get("linear_solver")
    if requested not in ("normal_cholmod", "normal_cholmod_aat"):
        return False
    if actual is None:
        # 早退或结构回退构造中断均可能尚未写入实际后端标签。
        return True
    if actual == requested:
        return report.get("ipm_structural_fallbacks") == 0
    return actual == "factorhighs" and type(report.get("ipm_structural_fallbacks")) is int and report["ipm_structural_fallbacks"] > 0


def valid_process(row):
    report = row.get("report")
    if row.get("error") is not None or not isinstance(report, dict) or type(row.get("returncode")) is not int:
        return False
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        return False
    known = {"optimal": (0, "resolved"), "empty": (0, "resolved"), "infeasible": (0, "resolved"),
        "unbounded": (0, "resolved"), "verification_failed": (3, "verification_failed"),
        "iteration_limit": (4, "incomplete"), "time_limit": (4, "incomplete"), "numerical_error": (4, "incomplete")}
    status = report.get("status")
    if not isinstance(status, str) or status not in known or (row["returncode"], row.get("classification")) != known[status]:
        return False
    if benchmark.classify(row["returncode"], report) != row["classification"] or row.get("status") != status:
        return False
    timings = report.get("timings")
    if not isinstance(timings, dict) or not all(benchmark.finite_real(timings.get(name)) and timings[name] >= 0
        for name in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")):
        return False
    backend = report.get("backend")
    options = report.get("options")
    if backend not in ("ipm", "highs") or not isinstance(options, dict):
        return False
    mode = None
    if backend == "ipm":
        ipm = options.get("ipm")
        if not isinstance(ipm, dict) or not isinstance(ipm.get("linear_accuracy"), str):
            return False
        mode = ipm["linear_accuracy"]
    if not valid_counts(report, None, backend, mode) or not valid_factor_path(report, backend):
        return False
    numerical, wall = row.get("numerical_seconds"), row.get("elapsed_wall_seconds")
    return benchmark.finite_real(numerical) and numerical >= 0 and benchmark.finite_real(wall) and wall > 0


def limited(row, frozen):
    return row.get("error") == "wall_timeout" or (row.get("report") or {}).get("status") == "time_limit" \
        or (benchmark.finite_real(row.get("elapsed_wall_seconds")) and row["elapsed_wall_seconds"] > frozen["wall_timeout"]) \
        or (benchmark.finite_real(row.get("numerical_seconds")) and row["numerical_seconds"] > frozen["time_limit"])


def pair_result(a, b, frozen):
    reports = [copy.deepcopy(row.get("report")) for row in (a, b)]
    for report in reports:
        if isinstance(report, dict):
            for field in frozen["compatibility_excluded_report_fields"]:
                report.pop(field, None)
    equal = typed_equal(*reports) and a.get("returncode") == b.get("returncode") and a.get("error") == b.get("error")
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, frozen) or limited(b, frozen)
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "valid_processes": valid, "budget_limited": budget, "category": category}


def validate_rows(stage, frozen, manifest, rows):
    expected = specification(stage, frozen)
    reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
    scheduled = list(compare_lp.schedule(expected["models"], expected["repeats"], (reference, candidate)))
    assert len(scheduled) == len(rows)
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, scheduled)):
        side = "reference" if variant == reference else "candidate"
        backend = "highs" if stage == "highs" and side == "reference" else "ipm"
        version = 57 if side == "reference" and stage != "highs" else 66
        solver = manifest["solvers"][variant]
        assert row["run_index"] == index and row["model_index"] == model["model_index"] and row["repetition"] == repeat
        assert row["position_in_pair"] == position and row["requested_variant"] == variant
        assert row["input"] == model["input"] and row["seed"] == frozen["seed"] and row["requested_backend"] == backend
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert all(row[field] == solver["sha256"] for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert row["hash_error"] is None
        # 墙钟杀进程也保留原始记录，但不能通过兼容合同或继续扩集。
        assert valid_process(row), "非法进程/状态记录：" + str(index)
        report = row["report"]
        assert report == benchmark.strict_json(row["stdout"])
        assert report["input"] == model["input"] and report["backend"] == backend
        expected_version = "1.15.1" if backend == "highs" else frozen[side + "_backend_version"]
        assert report["backend_version"] == expected_version
        assert typed_equal(report["options"], expected_options(stage, side, frozen))
        assert valid_counts(report, version, backend, accuracy(stage, side, frozen)), "既有IPM计数或常数恢复账本非法"
        args = row["command"]
        assert args[:3] == [solver["path"], "solve", model["input"]] and len(args[3:]) % 2 == 0
        flags = dict(zip(args[3::2], args[4::2]))
        assert len(flags) * 2 == len(args[3:])
        expected_flags = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
        if backend == "ipm":
            expected_flags.update({"--iteration-limit": "200", "--scaling-passes": "5", "--ipm-method": "homogeneous",
                "--ipm-linear-accuracy": accuracy(stage, side, frozen), "--ipm-linear-solver": linear_solver(stage, side, frozen),
                "--ipm-direction-retries": "0", "--ipm-objective-scaling": "off"})
            expected_flags.update({"--ipm-" + key: value for key, value in frozen["shared_ipm"].items()})
        else:
            expected_flags.update({"--highs-" + key: value for key, value in frozen["highs_reference"].items()})
        assert flags == expected_flags
        numerical = compare_lp.numerical_seconds(row)
        assert row["numerical_seconds"] == numerical
        failure = compare_lp.quality_failure(row, model, row["model_sha256_after"], solver["sha256"], row["solver_sha256_after"])
        wall_exceeded = row["elapsed_wall_seconds"] > frozen["wall_timeout"]
        numerical_exceeded = numerical > frozen["time_limit"]
        if failure is None and wall_exceeded:
            failure = "wall_budget_exceeded"
        if failure is None and numerical_exceeded:
            failure = "numerical_budget_exceeded"
        assert row["quality_failure"] == failure and row["quality_pass"] is (failure is None)
        assert row["wall_budget_exceeded"] is wall_exceeded and row["numerical_budget_exceeded"] is numerical_exceeded
        assert row["quality_penalized_wall_seconds"] == (min(row["elapsed_wall_seconds"], frozen["wall_timeout"]) if failure is None else frozen["wall_timeout"])
        assert row["quality_penalized_numerical_seconds"] == (min(numerical, frozen["time_limit"]) if failure is None else frozen["time_limit"])


def compatibility():
    frozen = plan()
    pairs = []
    for stage in COMPAT:
        manifest, _, rows = archive(stage)
        for model in specification(stage, frozen)["models"]:
            selected = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
            assert len(selected) == 2
            a, b = (selected[manifest[side + "_variant"]] for side in ("reference", "candidate"))
            pairs.append({"stage": stage, "input": model["input"], **pair_result(a, b, frozen)})
    assert len(pairs) == 17
    return {"passed": all(pair["category"] == "equal" for pair in pairs), "runs": 34, "pairs": pairs,
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "plan_sha256": PLAN_SHA}


def save_once(path, value):
    if path.exists():
        assert read(path) == value, "禁止覆盖不一致证据"
    else:
        save(path, value)


def validation_self_check():
    frozen = {"time_limit": 5, "wall_timeout": 35,
        "compatibility_excluded_report_fields": ["timings", "backend_version"]}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error",
        "report": {"schema_version": 1, "status": "numerical_error", "backend_version": "ipm-cpu-0.47.0",
            "backend": "ipm", "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod"}},
            **dict.fromkeys(COUNT_FIELDS, 0), "ipm_factor_backend": None,
            "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}},
        "numerical_seconds": 1, "elapsed_wall_seconds": 2}
    assert pair_result(row, row, frozen)["category"] == "equal"
    modern = copy.deepcopy(row); modern["report"]["backend_version"] = "ipm-cpu-0.53.0"
    assert pair_result(row, modern, frozen)["category"] == "equal"
    time_row = copy.deepcopy(row); time_row["report"]["status"] = time_row["status"] = "time_limit"
    assert pair_result(row, time_row, frozen)["category"] == "budget_limited"
    changed = copy.deepcopy(row); changed["report"]["extra"] = 1
    assert pair_result(row, changed, frozen)["category"] == "different"
    broken = []
    for key, value in (("returncode", False), ("returncode", 0), ("returncode", -1073741819), ("error", "wall_timeout"),
        ("error", "launch_error"), ("numerical_seconds", None), ("numerical_seconds", float("nan")), ("elapsed_wall_seconds", -1)):
        sample = copy.deepcopy(row); sample[key] = value; broken.append(sample)
    for status in ("bogus", None, "error"):
        sample = copy.deepcopy(row); sample["report"]["status"] = status; broken.append(sample)
    sample = copy.deepcopy(row); sample["report"]["schema_version"] = True; broken.append(sample)
    for sample in broken:
        assert pair_result(sample, time_row, frozen)["category"] == "invalid_process"
    base = dict.fromkeys(COUNT_FIELDS, 0)
    count_cases = [(base, "ipm", "homogeneous_repair", True),
        (dict.fromkeys(COUNT_FIELDS, None), "highs", None, True),
        (base, "highs", None, False)]
    for field in COUNT_FIELDS:
        for invalid in (True, False, -1, 1.0, 1 << 63):
            value = dict(base); value[field] = invalid
            count_cases.append((value, "ipm", "homogeneous_repair", False))
        value = dict(base); value.pop(field)
        count_cases.append((value, "ipm", "homogeneous_repair", False))
        value = dict(base); value[field] = None
        count_cases.append((value, "ipm", "homogeneous_repair", field in OPTIONAL_COUNT_FIELDS))
    for field in ABSENT_FIELDS:
        value = dict(base); value[field] = 0
        count_cases.append((value, "ipm", "homogeneous_repair", False))
    def work(weighted, refinements, repairs, successes, constants):
        return {**base, "ipm_weighted_krylov_steps": weighted, "ipm_refinement_steps": refinements,
                "ipm_homogeneous_constant_repairs": repairs, "ipm_homogeneous_constant_repair_successes": successes,
                "ipm_homogeneous_constant_solves": constants}
    count_cases += [(work(0, 8, 2, 1, 3), "ipm", "homogeneous_repair", True),
        (work(0, 8, 2, 3, 3), "ipm", "homogeneous_repair", False),
        (work(0, 8, 3, 0, 2), "ipm", "homogeneous_repair", False),
        (work(1, 1, 1, 1, 1), "ipm", "homogeneous_repair", False),
        (work(2, 1, 0, 0, 0), "ipm", "blockwise_weighted", False),
        (work(2, 3, 0, 0, 0), "ipm", "blockwise_weighted", True)]
    for report, backend, mode, expected in count_cases:
        assert valid_counts(report, 66, backend, mode) is expected
    all_status = budget_cases = 0
    for status, code, classification in (("optimal", 0, "resolved"), ("empty", 0, "resolved"),
        ("infeasible", 0, "resolved"), ("unbounded", 0, "resolved"),
        ("verification_failed", 3, "verification_failed"), ("iteration_limit", 4, "incomplete"),
        ("time_limit", 4, "incomplete"), ("numerical_error", 4, "incomplete")):
        for value, backend, mode, expected in count_cases:
            if backend != "ipm" or expected:
                continue
            bad = copy.deepcopy(row)
            bad.update(status=status, returncode=code, classification=classification)
            bad["report"].update(status=status, backend_version="ipm-cpu-0.53.0")
            bad["report"]["options"]["ipm"]["linear_accuracy"] = mode
            for field in COUNT_FIELDS:
                bad["report"].pop(field)
            bad["report"].update(value)
            assert not valid_counts(bad["report"], 66, backend, mode); all_status += 1
            assert pair_result(bad, time_row, frozen)["category"] == "invalid_process"; budget_cases += 1
    factor_cases = 0
    for requested in ("normal_cholmod", "normal_cholmod_aat"):
        for actual, fallbacks, expected in ((None, 0, True), (None, 1, True), (requested, 0, True),
            (requested, 1, False), ("factorhighs", 1, True), ("factorhighs", 0, False),
            ("normal_cholmod" if requested == "normal_cholmod_aat" else "normal_cholmod_aat", 0, False)):
            sample = copy.deepcopy(row["report"])
            sample["options"]["ipm"]["linear_solver"] = requested
            sample.update(ipm_factor_backend=actual, ipm_structural_fallbacks=fallbacks)
            assert valid_factor_path(sample, "ipm") is expected; factor_cases += 1
    assert not typed_equal({"x": True}, {"x": 1}) and typed_equal({"x": 1.0}, {"x": 1})
    return {"trajectory_cases": 4 + len(broken), "count_cases": len(count_cases),
        "all_status_count_cases": all_status, "invalid_budget_cases": budget_cases, "factor_path_cases": factor_cases, "typed_comparison_cases": 2}


def gate_decision(development, probe):
    def lost(summary):
        r, c = summary["reference_variant"], summary["candidate_variant"]
        return [Path(item["input"]).stem for item in summary["models"]
                if item["quality_pass_counts"][c] < item["quality_pass_counts"][r]]
    dev, pro = lost(development), lost(probe)
    r, c = development["reference_variant"], development["candidate_variant"]
    a, b = development["variants"][r], development["variants"][c]
    speed = b["penalized_numerical_sgm10_seconds"] < a["penalized_numerical_sgm10_seconds"] and b["penalized_wall_sgm10_seconds"] <= a["penalized_wall_sgm10_seconds"]
    pr, pc = probe["reference_variant"], probe["candidate_variant"]
    gain = [Path(item["input"]).stem for item in probe["models"] if item["quality_pass_counts"][pc] > item["quality_pass_counts"][pr]]
    return {"development_lost": dev, "probe_lost": pro, "probe_gain": gain,
            "development_speed_gate": speed, "eligible_for_netlib": not dev and not pro and (speed or bool(gain)),
            "plan_sha256": PLAN_SHA}


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    return gate_decision(development, probe)


def save_gate(decision):
    destination = DATA / "v66-gate.json"
    if destination.exists():
        assert read(destination) == decision, "已有门槛证据不一致，禁止覆盖"
    else:
        save(destination, decision)


def run(stage):
    frozen = plan()
    check_solver(57)
    check_solver(66)
    if stage not in COMPAT:
        compatible = compatibility()
        save_once(DATA / "v66-compatibility.json", compatible)
        assert compatible["passed"], "17个非预算完整报告对子未全部证明兼容，停止性能实验"
    if stage in ("netlib", "highs"):
        decision = gate()
        save_gate(decision)
        assert decision["eligible_for_netlib"], "逐例覆盖或开发成本未通过事前门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v66-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v66-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v66-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    identity = read(DATA / "ipm-v66-identity.json")
    assert all(digest == identity["files"]["scripts/" + Path(path).name] for path, digest in protected.items())
    with (DATA / f"v66-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v66-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v66-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v66-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "reference_solver_sha256_after": sha(EXE if stage == "highs" else PARENT),
        "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "removed_environment": ["OPENBLAS_CORETYPE", "所有SOLVER_V前缀变量"]})
    check_solver(57)
    check_solver(66)
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def self_check():
    frozen = plan()
    for stage in STAGES:
        command = command_for(stage, frozen, ROOT / "unused-v66-self-check")
        pairs = command[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert len(flags) * 2 == len(pairs)
        assert flags["--solver"] == str(EXE) and flags["--reference-solver"] == str(EXE if stage == "highs" else PARENT)
        assert flags["--ipm-linear-accuracy"] == accuracy(stage, "candidate", frozen) and flags["--ipm-objective-scaling"] == "off"
        assert flags["--ipm-linear-solver"] == linear_solver(stage, "candidate", frozen)
        assert expected_options(stage, "candidate", frozen)["ipm"]["linear_solver"] == flags["--ipm-linear-solver"]
        assert flags["--candidate-backend"] == "ipm" and flags["--reference-backend"] == ("highs" if stage == "highs" else "ipm")
        if stage == "highs":
            assert not any(key.startswith("--reference-ipm-") for key in flags)
            assert {key: flags["--reference-highs-" + key] for key in frozen["highs_reference"]} == frozen["highs_reference"]
        else:
            assert flags["--reference-ipm-linear-accuracy"] == "homogeneous_repair" and flags["--reference-ipm-objective-scaling"] == "off"
            assert flags["--reference-ipm-linear-solver"] == linear_solver(stage, "reference", frozen)
            assert expected_options(stage, "reference", frozen)["ipm"]["linear_solver"] == flags["--reference-ipm-linear-solver"]
    configuration_cases = 0
    for reference, candidate in (("strict", "homogeneous_repair"), ("homogeneous_repair", "strict")):
        altered = copy.deepcopy(frozen)
        altered.update(reference_accuracy=reference, candidate_accuracy=candidate, reference_objective_scaling="on", candidate_objective_scaling="off")
        pairs = command_for("development", altered, ROOT / "unused-v66-self-check")[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert flags["--reference-ipm-linear-accuracy"] == reference and flags["--ipm-linear-accuracy"] == candidate
        assert flags["--reference-ipm-objective-scaling"] == "on" and flags["--ipm-objective-scaling"] == "off"
        configuration_cases += 1
    for reference, candidate in (("normal_cholmod_aat", "normal_cholmod"), ("normal_cholmod", "normal_cholmod_aat")):
        altered = copy.deepcopy(frozen)
        altered.update(reference_linear_solver=reference, candidate_linear_solver=candidate)
        for stage in STAGES:
            pairs = command_for(stage, altered, ROOT / "unused-v66-self-check")[2:]
            flags = dict(zip(pairs[::2], pairs[1::2]))
            wanted = "normal_cholmod" if stage in COMPAT else candidate
            assert flags["--ipm-linear-solver"] == wanted
            assert expected_options(stage, "candidate", altered)["ipm"]["linear_solver"] == wanted
            if stage != "highs":
                wanted = "normal_cholmod" if stage in COMPAT else reference
                assert flags["--reference-ipm-linear-solver"] == wanted
                assert expected_options(stage, "reference", altered)["ipm"]["linear_solver"] == wanted
        configuration_cases += 1
    def sample(a, b, numerical=1, wall=2):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": a, "c": b}}],
            "variants": {"r": {"penalized_numerical_sgm10_seconds": 2, "penalized_wall_sgm10_seconds": 2},
                "c": {"penalized_numerical_sgm10_seconds": numerical, "penalized_wall_sgm10_seconds": wall}}}
    cases = [(3, 3, 0, 0, 1, 2, True), (3, 2, 0, 3, 1, 1, False), (3, 3, 3, 2, 1, 1, False),
             (3, 3, 0, 3, 3, 3, True), (3, 3, 0, 0, 3, 1, False), (3, 3, 0, 0, 1, 3, False),
             (3, 3, 0, 0, 2, 2, False), (3, 3, 0, 0, 1, 1, True), (0, 0, 0, 1, 2, 2, True),
             (3, 3, 0, 0, 1, 2.000000001, False)]
    for a, b, c, d, numerical, wall, expected in cases:
        assert gate_decision(sample(a, b, numerical, wall), sample(c, d))["eligible_for_netlib"] is expected
    return {"passed": True, "command_cases": len(STAGES), "configuration_cases": configuration_cases,
            "gate_cases": len(cases), **validation_self_check(), "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", *STAGES, "gate", "compatibility"))
    stage = parser.parse_args().stage
    if stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif stage == "compatibility":
        result = compatibility()
        save_once(DATA / "v66-compatibility.json", result)
        print(json.dumps(result, ensure_ascii=False))
    elif stage == "gate":
        result = gate()
        save_gate(result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(stage)
