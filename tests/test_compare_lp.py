#!/usr/bin/env python3
"""验证冻结对照的执行顺序、质量门槛、输入一致性和失败统计。"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


COMPARE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(COMPARE.parent))
sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("solver_compare_lp", COMPARE)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def report(backend):
    return {"schema_version": 1, "backend": backend, "status": "optimal", "has_solution": True,
            "options": {"pdlp": {"mode": "baseline", "history_size": 8, "max_iterations": 200000, "scaling_passes": 5} if backend == "pdlp" else None,
                        "ipm": {"method": "mehrotra", "max_iterations": 200, "scaling_passes": 5, "eliminate_singletons": False,
                                "dynamic_regularization": False, "equilibrate_kkt": False, "objective_scaling": True,
                                "linear_accuracy": "strict", "central_corrections": 0, "direction_retries": 0, "krylov_history_size": 0, "linear_solver": "qdldl",
                                "inexact_prefilter": False, "eliminate_fixed_variables": False,
                                "presolve_singleton_rows": False, "eliminate_presolved_fixed_variables": False} if backend == "ipm" else None,
                        "highs": {"solver": "choose", "run_crossover": "on", "presolve": "choose"} if backend == "highs" else None},
            "objective": 1.0, "solution": [1.0], "row_dual": [1.0], "best_bound": None,
            "ipm_homogeneous_constant_solves": 0 if backend == "ipm" else None,
            "ipm_homogeneous_constant_repairs": 0 if backend == "ipm" else None,
            "ipm_homogeneous_constant_repair_successes": 0 if backend == "ipm" else None,
            "ipm_homogeneous_direction_solves": 0 if backend == "ipm" else None,
            "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
            "ipm_homogeneous_scaled_products": 0 if backend == "ipm" else None,
            "ipm_homogeneous_scaled_refinement_steps": 0 if backend == "ipm" else None,
            "ipm_homogeneous_scaled_inexact_returns": 0 if backend == "ipm" else None,
            "model": {"columns": 1, "rows": 1, "is_mip": False},
            "verification": {"valid_input": True, "feasible": True, "objective_consistent": True,
                             "recomputed_objective": 1.0},
            "lp_optimality": {"valid_input": True, "dual_feasible": True, "gap_satisfied": True,
                              "numerically_optimal": True, "max_dual_residual": 0.0,
                              "dual_objective_estimate": 1.0, "objective_gap": 0.0,
                              "relative_objective_gap": 0.0},
            "timings": {"read_seconds": 0.01, "solve_seconds": 0.1,
                        "verification_seconds": 0.01, "total_seconds": 0.12}}


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows 的首次平台探测可能启动系统命令，须在模拟求解子进程之前完成。
        COMPARISON.platform.platform()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="solver comparison ")
        self.root = Path(self.temporary.name)
        self.solver = self.root / "solver placeholder.exe"
        self.solver.write_bytes(b"frozen solver")
        self.models = [self.root / "model one.lp", self.root / "model two.lp"]
        for model in self.models:
            model.write_text("Minimize\n obj: x\nBounds\n x >= 1\nEnd\n", encoding="utf-8")
        self.model_list = self.root / "models.txt"
        self.model_list.write_text("\n".join(model.name for model in self.models) + "\n", encoding="utf-8")
        self.output = self.root / "output"
        self.commands = []

    def tearDown(self):
        self.temporary.cleanup()

    def fake_process(self, command, **kwargs):
        self.commands.append(command)
        backend = command[command.index("--backend") + 1]
        value = report(backend)
        if backend == "pdlp":
            value["options"]["pdlp"]["mode"] = command[command.index("--pdlp-mode") + 1]
            if "--pdlp-history-size" in command:
                value["options"]["pdlp"]["history_size"] = int(command[command.index("--pdlp-history-size") + 1])
        if backend in {"ipm", "pdlp"}:
            for flag, field in [("--iteration-limit", "max_iterations"), ("--scaling-passes", "scaling_passes")]:
                if flag in command:
                    value["options"][backend][field] = int(command[command.index(flag) + 1])
        if backend == "ipm":
            if "--ipm-method" in command:
                value["options"]["ipm"]["method"] = command[command.index("--ipm-method") + 1]
            if value["options"]["ipm"]["method"] == "homogeneous":
                value.update(ipm_homogeneous_constant_solves=3, ipm_homogeneous_direction_solves=6,
                             ipm_homogeneous_tau=1.5, ipm_homogeneous_kappa=1e-8)
            for flag, field in (("--ipm-singleton-elimination", "eliminate_singletons"),
                                ("--ipm-dynamic-regularization", "dynamic_regularization"),
                                ("--ipm-kkt-equilibration", "equilibrate_kkt"),
                                ("--ipm-objective-scaling", "objective_scaling"),
                                ("--ipm-inexact-prefilter", "inexact_prefilter"),
                                ("--ipm-fixed-variable-elimination", "eliminate_fixed_variables"),
                                ("--ipm-singleton-presolve", "presolve_singleton_rows"),
                                ("--ipm-presolve-fixed-elimination", "eliminate_presolved_fixed_variables")):
                if flag in command:
                    value["options"]["ipm"][field] = command[command.index(flag) + 1] == "on"
            enabled = value["options"]["ipm"]["eliminate_singletons"]
            if "--ipm-linear-accuracy" in command:
                value["options"]["ipm"]["linear_accuracy"] = command[command.index("--ipm-linear-accuracy") + 1]
            if value["options"]["ipm"]["linear_accuracy"] == "homogeneous_scaled":
                value.update(ipm_homogeneous_scaled_products=10, ipm_homogeneous_scaled_refinement_steps=4,
                             ipm_homogeneous_scaled_inexact_returns=2, ipm_homogeneous_direction_solves=8)
            if value["options"]["ipm"]["linear_accuracy"] == "homogeneous_repair":
                value.update(ipm_homogeneous_constant_repairs=2, ipm_homogeneous_constant_repair_successes=1,
                             ipm_homogeneous_constant_solves=5)
            for flag, field in (("--ipm-central-corrections", "central_corrections"),
                                ("--ipm-direction-retries", "direction_retries"),
                                ("--ipm-krylov-history-size", "krylov_history_size")):
                if flag in command:
                    value["options"]["ipm"][field] = int(command[command.index(flag) + 1])
            if "--ipm-linear-solver" in command:
                value["options"]["ipm"]["linear_solver"] = command[command.index("--ipm-linear-solver") + 1]
            history = value["options"]["ipm"]["krylov_history_size"]
            value.update(ipm_krylov_recycle_projections=2 if history else 0,
                         ipm_krylov_recycle_acceptances=1 if history else 0,
                         ipm_krylov_recycle_vectors=history,
                         ipm_krylov_recycle_peak_dimension=history)
            full_newton = value["options"]["ipm"]["linear_accuracy"] in ("full_newton", "bounded_newton")
            bounded = value["options"]["ipm"]["linear_accuracy"] == "bounded_newton"
            value.update(ipm_bounded_newton_returns=4 if bounded else 0,
                         ipm_bounded_newton_inexact_returns=2 if bounded else 0)
            value.update(ipm_full_newton_preconditioner_solves=6 if full_newton else 0,
                         ipm_full_newton_products=9 if full_newton else 0)
            value.update(ipm_kkt_dimension=2 if enabled else 3, ipm_factor_nonzeros=3 if enabled else 5,
                         ipm_eliminated_singletons=1 if enabled else 0,
                         ipm_eliminated_fixed_variables=int(value["options"]["ipm"]["eliminate_fixed_variables"]) + int(value["options"]["ipm"]["eliminate_presolved_fixed_variables"]),
                         ipm_presolved_fixed_columns=1 if value["options"]["ipm"]["eliminate_presolved_fixed_variables"] else 0,
                         ipm_presolved_singleton_rows=2 if value["options"]["ipm"]["presolve_singleton_rows"] else 0,
                         ipm_regularized_pivots=2 if value["options"]["ipm"]["dynamic_regularization"] else 0,
                         ipm_backward_error_acceptances=3 if value["options"]["ipm"]["linear_accuracy"] == "backward" else 0,
                         ipm_inexact_acceptances=5 if value["options"]["ipm"]["linear_accuracy"] == "inexact" else 0,
                         ipm_inexact_prefilter_rejections=7 if value["options"]["ipm"]["inexact_prefilter"] else 0,
                         ipm_central_correction_attempts=5 if value["options"]["ipm"]["central_corrections"] else 0,
                         ipm_central_correction_acceptances=2 if value["options"]["ipm"]["central_corrections"] else 0)
        if backend == "highs":
            for flag, field in [("--highs-solver", "solver"), ("--highs-crossover", "run_crossover"),
                                ("--highs-presolve", "presolve")]:
                if flag in command:
                    value["options"]["highs"][field] = command[command.index(flag) + 1]
        return subprocess.CompletedProcess(command, 0, json.dumps(value).encode("utf-8"), b"")

    def run_comparison(self, responder=None, repeats=3, extra_args=(), default_pdlp=True):
        args = ["--solver", str(self.solver), "--models-file", str(self.model_list),
                "--output-dir", str(self.output), "--repeats", str(repeats), "--time-limit", "2",
                "--wall-timeout", "3", "--threads", "1", "--seed", "7"]
        if default_pdlp:
            args.extend(["--pdlp-mode", "baseline"])
        args.extend(extra_args)
        with mock.patch.object(COMPARISON.benchmark.subprocess, "run", side_effect=responder or self.fake_process), \
                contextlib.redirect_stdout(io.StringIO()):
            status = COMPARISON.main(args)
        records = [json.loads(line) for line in (self.output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        return status, records, summary

    def test_independent_binaries_alternate_with_identical_options(self):
        reference_solver = self.root / "reference solver.exe"
        reference_solver.write_bytes(b"reference binary")
        status, records, summary = self.run_comparison(repeats=1, default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm",
            "--reference-solver", str(reference_solver)])
        self.assertEqual(status, 0)
        self.assertEqual([command[0] for command in self.commands],
                         list(map(str, [reference_solver, self.solver, self.solver, reference_solver])))
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        for record in records:
            expected = manifest["solvers"][record["requested_variant"]]
            self.assertEqual(record["command"][0], expected["path"])
            self.assertEqual(record["solver_sha256_before"], expected["sha256"])
            self.assertEqual(record["frozen_solver_sha256"], expected["sha256"])
        self.assertTrue(summary["reference_variant"].endswith(":binary=reference"))
        self.assertTrue(summary["candidate_variant"].endswith(":binary=candidate"))

    def test_reference_binary_changes_only_fail_its_own_records(self):
        reference_solver = self.root / "reference solver.exe"
        reference_solver.write_bytes(b"reference binary")

        def responder(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            if command[0] == str(reference_solver):
                reference_solver.write_bytes(b"changed reference binary")
            return process

        status, records, summary = self.run_comparison(responder, repeats=1, default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm",
            "--reference-solver", str(reference_solver)])
        self.assertEqual(status, 1)
        self.assertEqual(summary["quality_pass_count"], 2)
        for record in records:
            expected = "solver_changed_or_unavailable" if record["command"][0] == str(reference_solver) else None
            self.assertEqual(record["quality_failure"], expected)

    def test_unavailable_reference_binary_cannot_pass(self):
        reference_solver = self.root / "missing.exe"
        status, records, summary = self.run_comparison(repeats=1, default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm",
            "--reference-solver", str(reference_solver)])
        self.assertEqual(status, 1)
        self.assertEqual(summary["quality_pass_count"], 2)
        self.assertTrue(all(record["quality_failure"] == "solver_changed_or_unavailable"
                            for record in records if record["command"][0] == str(reference_solver)))

    def test_fixed_highs_point_reference_and_mismatch(self):
        def responder(command, **kwargs):
            self.commands.append(command)
            backend = command[command.index("--backend") + 1]
            value = report(backend)
            if backend == "highs":
                self.assertEqual(command[command.index("--highs-solver") + 1], "ipm")
                self.assertEqual(command[command.index("--highs-crossover") + 1], "off")
                value["options"]["highs"] = {"solver": "ipm", "run_crossover": "off", "presolve": "choose"}
                if len(self.commands) == 1:
                    value["options"]["highs"]["run_crossover"] = "on"
            else:
                self.assertNotIn("--highs-solver", command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(value).encode(), stderr=b"")
        status, records, summary = self.run_comparison(responder=responder, repeats=1,
            extra_args=("--reference-highs-solver", "ipm", "--reference-highs-crossover", "off"))
        self.assertEqual(status, 1)
        self.assertEqual(records[0]["classification"], "highs_crossover_mismatch")
        self.assertFalse(records[0]["quality_pass"])
        self.assertEqual(summary["quality_pass_count"], 3)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        label = "highs:solver=ipm:crossover=off:presolve=choose"
        self.assertEqual(manifest["variants"][label]["highs_solver"], "ipm")
        self.assertEqual(manifest["variants"][label]["highs_crossover"], "off")
        self.assertEqual(summary["reference_variant"], label)
        self.assertTrue(all(model["wall_speedup_highs_over_pdlp"] is None for model in summary["models"]))

    def test_highs_options_reject_wrong_backend(self):
        arguments = ["--solver", str(self.solver), "--models-file", str(self.model_list)]
        for extra in [("--highs-solver", "ipm"), ("--highs-presolve", "off"),
                      ("--reference-backend", "ipm", "--reference-highs-crossover", "off"),
                      ("--reference-backend", "ipm", "--reference-highs-presolve", "off")]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                COMPARISON.main(arguments + list(extra))
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.commands)

    def test_highs_presolve_ablation_keeps_configurations_separate(self):
        status, records, summary = self.run_comparison(repeats=1, default_pdlp=False, extra_args=[
            "--candidate-backend", "highs", "--highs-presolve", "on", "--reference-highs-presolve", "off"])
        self.assertEqual(status, 0)
        reference = "highs:solver=choose:crossover=on:presolve=off"
        candidate = "highs:solver=choose:crossover=on:presolve=on"
        self.assertEqual(set(summary["variants"]), {reference, candidate})
        self.assertIsNone(summary["backends"])
        for record in records:
            expected = "off" if record["requested_variant"] == reference else "on"
            self.assertEqual(record["command"][record["command"].index("--highs-presolve") + 1], expected)
            self.assertEqual(record["report"]["options"]["highs"]["presolve"], expected)

    def test_highs_presolve_mismatch_fails_and_does_not_reach_candidate(self):
        def responder(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "highs":
                self.assertEqual(command[command.index("--highs-presolve") + 1], "off")
                value["options"]["highs"]["presolve"] = "choose"
                process.stdout = json.dumps(value).encode()
            else:
                self.assertNotIn("--highs-presolve", command)
            return process
        status, records, summary = self.run_comparison(responder, repeats=1,
                                                      extra_args=["--reference-highs-presolve", "off"])
        self.assertEqual(status, 1)
        self.assertEqual(summary["variants"][summary["reference_variant"]]["quality_failure_counts"],
                         {"highs_presolve_mismatch": 2})
        self.assertTrue(all(record["quality_pass"] for record in records if record["requested_backend"] == "pdlp"))

    def test_alternating_order_and_identical_budgets(self):
        status, records, summary = self.run_comparison()
        self.assertEqual(status, 0)
        self.assertEqual(len(records), 12)
        expected = ["highs", "pdlp", "pdlp", "highs", "highs", "pdlp",
                    "pdlp", "highs", "highs", "pdlp", "pdlp", "highs"]
        self.assertEqual([record["requested_backend"] for record in records], expected)
        for index, command in enumerate(self.commands):
            for name, value in [("--time-limit", "2.0"), ("--threads", "1"), ("--seed", "7")]:
                self.assertEqual(command[command.index(name) + 1], value)
            if expected[index] == "pdlp":
                self.assertEqual(command[command.index("--pdlp-mode") + 1], "baseline")
            else:
                self.assertNotIn("--pdlp-mode", command)
        self.assertEqual(summary["both_backends_all_repeats_pass_count"], 2)
        self.assertEqual(summary["quality_pass_count"], 12)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["models_file_sha256"], COMPARISON.benchmark.sha256(self.model_list))
        for record in records:
            self.assertEqual(record["model_sha256"], record["frozen_model_sha256"])

    def test_quality_failure_and_timeout_remain_in_all_statistics(self):
        calls = 0

        def respond(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise subprocess.TimeoutExpired(command, 3, output=b"partial")
            process = self.fake_process(command, **kwargs)
            if calls == 1:
                value = json.loads(process.stdout)
                value["lp_optimality"]["dual_feasible"] = False
                process.stdout = json.dumps(value).encode("utf-8")
            return process

        status, records, summary = self.run_comparison(respond, repeats=1)
        self.assertEqual(status, 1)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["quality_failure"], "optimality_quality_failed")
        self.assertEqual(records[1]["quality_failure"], "wall_timeout")
        self.assertEqual(records[1]["stdout"], "partial")
        self.assertEqual(records[0]["quality_penalized_wall_seconds"], 3)
        self.assertEqual(records[1]["quality_penalized_numerical_seconds"], 2)
        self.assertIsNone(summary["models"][0]["wall_speedup_highs_over_pdlp"])
        self.assertIsNone(summary["models"][0]["wall_winner"])
        self.assertEqual(summary["both_backends_all_repeats_pass_count"], 1)
        self.assertEqual(summary["backends"]["highs"]["quality_pass_count"], 1)
        self.assertEqual(summary["backends"]["pdlp"]["quality_pass_count"], 1)

    def test_missing_input_is_not_replaced_or_dropped(self):
        self.models[0].unlink()
        status, records, summary = self.run_comparison(repeats=1)
        self.assertEqual(status, 1)
        self.assertEqual(len(records), 4)
        self.assertEqual([record["quality_failure"] for record in records[:2]],
                         ["input_unavailable_at_freeze", "input_unavailable_at_freeze"])
        self.assertEqual([Path(command[2]) for command in self.commands[:2]], [self.models[0]] * 2)
        self.assertEqual(summary["model_count"], 2)

    def test_modified_model_invalidates_comparison(self):
        changed = False

        def respond(command, **kwargs):
            nonlocal changed
            if not changed:
                Path(command[2]).write_text("changed model\n", encoding="utf-8")
                changed = True
            return self.fake_process(command, **kwargs)

        status, records, summary = self.run_comparison(respond, repeats=1)
        self.assertEqual(status, 1)
        self.assertEqual(records[0]["quality_failure"], "input_changed")
        self.assertEqual(records[1]["quality_failure"], "input_changed")
        self.assertIsNone(summary["models"][0]["wall_speedup_highs_over_pdlp"])

    def test_reported_pdlp_mode_must_match_requested_mode(self):
        def respond(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "pdlp":
                value["options"]["pdlp"]["mode"] = "adaptive"
                process.stdout = json.dumps(value).encode("utf-8")
            return process

        status, records, summary = self.run_comparison(respond, repeats=1)
        self.assertEqual(status, 1)
        self.assertEqual(summary["backends"]["pdlp"]["quality_failure_counts"], {"pdlp_mode_mismatch": 2})
        self.assertTrue(all(record["quality_pass"] for record in records if record["requested_backend"] == "highs"))

    def test_pdlp_ablation_keeps_modes_separate_and_alternates_order(self):
        status, records, summary = self.run_comparison(extra_args=[
            "--reference-pdlp-mode", "adaptive", "--reference-pdlp-history-size", "0",
            "--pdlp-mode", "subspace", "--pdlp-history-size", "8"])
        self.assertEqual(status, 0)
        reference, candidate = "pdlp:adaptive:history=0", "pdlp:subspace:history=8"
        self.assertTrue(all(record["requested_backend"] == "pdlp" for record in records))
        self.assertEqual([record["requested_variant"] for record in records],
                         [reference, candidate, candidate, reference, reference, candidate,
                          candidate, reference, reference, candidate, candidate, reference])
        self.assertEqual(set(summary["variants"]), {reference, candidate})
        self.assertEqual(summary["variants"][reference]["run_count"], 6)
        self.assertEqual(summary["variants"][candidate]["run_count"], 6)
        self.assertEqual(summary["both_variants_all_repeats_pass_count"], 2)
        self.assertIsNone(summary["backends"])
        for item in summary["models"]:
            self.assertIsNone(item["wall_speedup_highs_over_pdlp"])
            self.assertIsNotNone(item["wall_speedup_reference_over_candidate"])
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["reference_variant"], reference)
        self.assertEqual(manifest["candidate_variant"], candidate)
        self.assertEqual(manifest["variants"][reference]["history_size"], 0)
        for record in records:
            expected = 0 if record["requested_variant"] == reference else 8
            self.assertEqual(record["report"]["options"]["pdlp"]["history_size"], expected)

    def test_history_size_mismatch_is_not_successful(self):
        def respond(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "pdlp":
                value["options"]["pdlp"]["history_size"] = False
                process.stdout = json.dumps(value).encode("utf-8")
            return process

        status, records, summary = self.run_comparison(respond, repeats=1,
                                                      extra_args=["--pdlp-history-size", "0"])
        self.assertEqual(status, 1)
        self.assertEqual(summary["backends"]["pdlp"]["quality_failure_counts"], {"pdlp_history_size_mismatch": 2})

    def test_same_mode_with_different_history_sizes_is_a_valid_ablation(self):
        status, records, summary = self.run_comparison(repeats=1, extra_args=[
            "--reference-pdlp-mode", "subspace", "--reference-pdlp-history-size", "0",
            "--pdlp-mode", "subspace", "--pdlp-history-size", "32"])
        self.assertEqual(status, 0)
        self.assertEqual(set(summary["variants"]), {"pdlp:subspace:history=0", "pdlp:subspace:history=32"})

    def test_invalid_history_comparison_configuration_is_rejected(self):
        for flags in [["--reference-pdlp-history-size", "8"],
                      ["--reference-pdlp-mode", "baseline"],
                      ["--pdlp-history-size", "-1"], ["--pdlp-history-size", "33"],
                      ["--reference-pdlp-mode", "adaptive", "--reference-pdlp-history-size", "33"]]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                self.run_comparison(extra_args=flags)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.output.exists())

    def test_ipm_candidate_uses_independent_options_and_common_quality(self):
        status, records, summary = self.run_comparison(default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--iteration-limit", "123", "--scaling-passes", "0"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["reference_variant"], "highs")
        self.assertEqual(summary["candidate_variant"], "ipm")
        self.assertEqual(set(summary["backends"]), {"highs", "ipm"})
        self.assertTrue(all(item["wall_speedup_highs_over_pdlp"] is None for item in summary["models"]))
        for record in records:
            command = record["command"]
            self.assertNotIn("--pdlp-mode", command)
            self.assertNotIn("--pdlp-history-size", command)
            if record["requested_backend"] == "ipm":
                self.assertEqual(record["report"]["options"]["ipm"],
                                 {"method": "mehrotra", "max_iterations": 123, "scaling_passes": 0, "eliminate_singletons": False,
                                  "dynamic_regularization": False, "equilibrate_kkt": False, "objective_scaling": True, "linear_accuracy": "strict",
                                  "central_corrections": 0, "direction_retries": 0, "krylov_history_size": 0, "linear_solver": "qdldl", "inexact_prefilter": False,
                                  "eliminate_fixed_variables": False, "presolve_singleton_rows": False, "eliminate_presolved_fixed_variables": False})
                self.assertNotIn("--ipm-singleton-elimination", command)
                self.assertNotIn("--ipm-direction-retries", command)
                self.assertNotIn("--ipm-krylov-history-size", command)
                self.assertNotIn("--ipm-singleton-presolve", command)
                self.assertNotIn("--ipm-presolve-fixed-elimination", command)
                self.assertNotIn("--ipm-dynamic-regularization", command)
                self.assertNotIn("--ipm-kkt-equilibration", command)
                self.assertNotIn("--ipm-linear-accuracy", command)
                self.assertNotIn("--ipm-central-corrections", command)
                self.assertNotIn("--ipm-linear-solver", command)
                self.assertNotIn("--ipm-inexact-prefilter", command)
            else:
                self.assertNotIn("--iteration-limit", command)
                self.assertNotIn("--scaling-passes", command)

    def test_mixed_pdlp_reference_keeps_its_own_iteration_default(self):
        status, records, summary = self.run_comparison(repeats=1, default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--iteration-limit", "123", "--reference-pdlp-mode", "halpern"])
        self.assertEqual(status, 0)
        self.assertEqual(set(summary["backends"]), {"pdlp", "ipm"})
        for record in records:
            if record["requested_backend"] == "pdlp":
                self.assertNotIn("--iteration-limit", record["command"])
                self.assertEqual(record["report"]["options"]["pdlp"]["mode"], "halpern")
                self.assertEqual(record["report"]["options"]["pdlp"]["max_iterations"], 200000)

    def test_ipm_reference_accepts_explicit_separate_budget(self):
        status, records, summary = self.run_comparison(repeats=1, extra_args=[
            "--reference-backend", "ipm", "--reference-iteration-limit", "321",
            "--reference-scaling-passes", "0", "--iteration-limit", "12345", "--pdlp-mode", "halpern_reflected"])
        self.assertEqual(status, 0)
        for record in records:
            backend = record["requested_backend"]
            self.assertEqual(record["report"]["options"][backend]["max_iterations"], 321 if backend == "ipm" else 12345)
        self.assertEqual(summary["reference_variant"], "ipm")

    def test_ipm_singleton_ablation_has_independent_switches_and_statistics(self):
        status, records, summary = self.run_comparison(default_pdlp=False, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm",
            "--ipm-singleton-elimination", "on", "--iteration-limit", "123"])
        self.assertEqual(status, 0)
        reference = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
        candidate = "ipm:singletons=on:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
        self.assertEqual(set(summary["variants"]), {reference, candidate})
        self.assertEqual([record["requested_variant"] for record in records[:4]],
                         [reference, candidate, candidate, reference])
        self.assertIsNone(summary["backends"])
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest["reference_ipm_singleton_elimination"])
        self.assertEqual(manifest["ipm_singleton_elimination"], "on")
        for record in records:
            enabled = record["requested_variant"] == candidate
            self.assertIs(record["report"]["options"]["ipm"]["eliminate_singletons"], enabled)
            self.assertEqual(record["report"]["options"]["ipm"]["max_iterations"], 123)
            self.assertEqual(record["report"]["ipm_eliminated_singletons"], 1 if enabled else 0)
            self.assertEqual(record["report"]["ipm_kkt_dimension"], 2 if enabled else 3)
            self.assertEqual(record["report"]["ipm_factor_nonzeros"], 3 if enabled else 5)
            self.assertEqual("--ipm-singleton-elimination" in record["command"], enabled)
            self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_singleton_elimination"],
                             "on" if enabled else None)

    def test_ipm_switch_is_isolated_in_heterogeneous_comparisons(self):
        cases = [(["--candidate-backend", "ipm", "--ipm-singleton-elimination", "on"], False,
                  "--ipm-singleton-elimination", "on", "ipm:singletons=on:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-singleton-elimination", "off"], True,
                  "--ipm-singleton-elimination", "off", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-dynamic-regularization", "on"], False,
                  "--ipm-dynamic-regularization", "on", "ipm:singletons=off:dynamic=on:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-dynamic-regularization", "on"], True,
                  "--ipm-dynamic-regularization", "on", "ipm:singletons=off:dynamic=on:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-kkt-equilibration", "on"], False,
                  "--ipm-kkt-equilibration", "on", "ipm:singletons=off:dynamic=off:equilibrate=on:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-kkt-equilibration", "on"], True,
                  "--ipm-kkt-equilibration", "on", "ipm:singletons=off:dynamic=off:equilibrate=on:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-linear-accuracy", "backward"], False,
                  "--ipm-linear-accuracy", "backward", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=backward:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-linear-accuracy", "backward"], True,
                  "--ipm-linear-accuracy", "backward", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=backward:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-linear-accuracy", "inexact"], False,
                  "--ipm-linear-accuracy", "inexact", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=inexact:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-linear-accuracy", "inexact"], True,
                  "--ipm-linear-accuracy", "inexact", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=inexact:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-central-corrections", "0"], False,
                  "--ipm-central-corrections", "0", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-central-corrections", "1"], True,
                  "--ipm-central-corrections", "1", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=1:solver=qdldl:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-linear-solver", "sparselu"], False,
                  "--ipm-linear-solver", "sparselu", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=sparselu:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-linear-solver", "sparselu"], True,
                  "--ipm-linear-solver", "sparselu", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=sparselu:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-linear-solver", "normal", "--ipm-singleton-elimination", "on"], False,
                  "--ipm-linear-solver", "normal", "ipm:singletons=on:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=normal:prefilter=off:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-linear-solver", "normal", "--reference-ipm-singleton-elimination", "on"], True,
                  "--ipm-linear-solver", "normal", "ipm:singletons=on:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=normal:prefilter=off:fixed=off"),
                 (["--candidate-backend", "ipm", "--ipm-linear-accuracy", "inexact", "--ipm-inexact-prefilter", "on"], False,
                  "--ipm-inexact-prefilter", "on", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=inexact:corrections=0:solver=qdldl:prefilter=on:fixed=off"),
                 (["--reference-backend", "ipm", "--reference-ipm-inexact-prefilter", "off"], True,
                  "--ipm-inexact-prefilter", "off", "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off")]
        for index, (flags, default_pdlp, flag, expected, variant) in enumerate(cases):
            with self.subTest(flags=flags):
                self.output = self.root / f"heterogeneous-{index}"
                status, records, summary = self.run_comparison(repeats=1, default_pdlp=default_pdlp, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertIn(variant, summary["variants"])
                for record in records:
                    if record["requested_backend"] == "ipm":
                        self.assertEqual(record["command"][record["command"].index(flag) + 1], expected)
                    else:
                        self.assertNotIn("--ipm-singleton-elimination", record["command"])
                        self.assertNotIn("--ipm-dynamic-regularization", record["command"])
                        self.assertNotIn("--ipm-kkt-equilibration", record["command"])
                        self.assertNotIn("--ipm-linear-accuracy", record["command"])
                        self.assertNotIn("--ipm-central-corrections", record["command"])
                        self.assertNotIn("--ipm-linear-solver", record["command"])
                        self.assertNotIn("--ipm-inexact-prefilter", record["command"])

    def test_three_ipm_switches_form_independent_combined_variants(self):
        cases = [("off", "on", "off", "on", "off", "on"),
                 ("on", "off", "off", "on", "on", "off"),
                 ("on", "on", "off", "on", "on", "on")]
        for index, (reference_singleton, reference_dynamic, reference_equilibration,
                    singleton, dynamic, equilibration) in enumerate(cases):
            with self.subTest(case=index):
                self.output = self.root / f"combined-switches-{index}"
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
                    "--candidate-backend", "ipm", "--reference-backend", "ipm",
                    "--ipm-singleton-elimination", singleton, "--ipm-dynamic-regularization", dynamic,
                    "--ipm-kkt-equilibration", equilibration,
                    "--reference-ipm-singleton-elimination", reference_singleton,
                    "--reference-ipm-dynamic-regularization", reference_dynamic,
                    "--reference-ipm-kkt-equilibration", reference_equilibration])
                self.assertEqual(status, 0)
                reference = f"ipm:singletons={reference_singleton}:dynamic={reference_dynamic}:equilibrate={reference_equilibration}:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
                candidate = f"ipm:singletons={singleton}:dynamic={dynamic}:equilibrate={equilibration}:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
                self.assertEqual(set(summary["variants"]), {reference, candidate})
                self.assertIsNone(summary["backends"])
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_kkt_equilibration"], equilibration)
                self.assertEqual(manifest["reference_ipm_kkt_equilibration"], reference_equilibration)
                for record in records:
                    expected_singleton, expected_dynamic, expected_equilibration = (
                        (reference_singleton, reference_dynamic, reference_equilibration)
                        if record["requested_variant"] == reference else (singleton, dynamic, equilibration))
                    self.assertIs(record["report"]["options"]["ipm"]["eliminate_singletons"], expected_singleton == "on")
                    self.assertIs(record["report"]["options"]["ipm"]["dynamic_regularization"], expected_dynamic == "on")
                    self.assertIs(record["report"]["options"]["ipm"]["equilibrate_kkt"], expected_equilibration == "on")
                    self.assertEqual(record["report"]["ipm_regularized_pivots"], 2 if expected_dynamic == "on" else 0)
                    configuration = manifest["variants"][record["requested_variant"]]
                    self.assertEqual(configuration["ipm_singleton_elimination"], expected_singleton)
                    self.assertEqual(configuration["ipm_dynamic_regularization"], expected_dynamic)
                    self.assertEqual(configuration["ipm_kkt_equilibration"], expected_equilibration)

    def test_ipm_factor_options_wrong_boolean_or_missing_report_fails(self):
        for option, field in (("ipm-dynamic-regularization", "dynamic_regularization"),
                              ("ipm-kkt-equilibration", "equilibrate_kkt")):
            for index, reported in enumerate((1, "on", None, False)):
                with self.subTest(option=option, reported=reported):
                    self.output = self.root / f"bad-{option}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported is None:
                                del value["options"]["ipm"][field]
                            else:
                                value["options"]["ipm"][field] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=[
                        "--candidate-backend", "ipm", "--ipm-singleton-elimination", "on", "--" + option, "on"])
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["quality_pass_count"], 2)
                    self.assertEqual(summary["variants"][summary["candidate_variant"]]["quality_failure_counts"],
                                     {option.replace("-", "_") + "_mismatch": 2})

    def test_kkt_equilibration_is_not_copied_to_reference(self):
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-kkt-equilibration", "on"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["reference_variant"], "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off")
        self.assertEqual(summary["candidate_variant"], "ipm:singletons=off:dynamic=off:equilibrate=on:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off")
        for record in records:
            enabled = record["requested_variant"] == summary["candidate_variant"]
            self.assertIs(record["report"]["options"]["ipm"]["equilibrate_kkt"], enabled)
            self.assertEqual("--ipm-kkt-equilibration" in record["command"], enabled)

    def test_linear_accuracy_is_independent_and_preserves_acceptance_count(self):
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-linear-accuracy", "backward"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["reference_variant"], "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off")
        self.assertEqual(summary["candidate_variant"], "ipm:singletons=off:dynamic=off:equilibrate=off:linear=backward:corrections=0:solver=qdldl:prefilter=off:fixed=off")
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest["reference_ipm_linear_accuracy"])
        self.assertEqual(manifest["ipm_linear_accuracy"], "backward")
        for record in records:
            enabled = record["requested_variant"] == summary["candidate_variant"]
            self.assertEqual(record["report"]["options"]["ipm"]["linear_accuracy"], "backward" if enabled else "strict")
            self.assertEqual("--ipm-linear-accuracy" in record["command"], enabled)
            self.assertEqual(record["report"]["ipm_backward_error_acceptances"], 3 if enabled else 0)
            self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_linear_accuracy"],
                             "backward" if enabled else None)

    def test_ipm_methods_are_independent_and_preserve_actual_statistics(self):
        for side in ("candidate", "reference"):
            for explicit_default in (False, True):
                with self.subTest(side=side, explicit_default=explicit_default):
                    self.output = self.root / f"method-{side}-{explicit_default}"
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "--reference-" if side == "candidate" else "--"
                    flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                             prefix + "ipm-method", "homogeneous", prefix + "ipm-linear-accuracy", "strict",
                             prefix + "ipm-central-corrections", "0", prefix + "ipm-direction-retries", "0",
                             prefix + "ipm-krylov-history-size", "0", prefix + "ipm-inexact-prefilter", "off",
                             prefix + "ipm-linear-solver", "normal_augmented", prefix + "ipm-singleton-elimination", "on",
                             prefix + "ipm-kkt-equilibration", "on", prefix + "ipm-fixed-variable-elimination", "on",
                             other + "ipm-central-corrections", "1", other + "ipm-krylov-history-size", "1"]
                    if explicit_default:
                        flags += [other + "ipm-method", "mehrotra"]
                    status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 0)
                    manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                    requested_key = "ipm_method" if side == "candidate" else "reference_ipm_method"
                    self.assertEqual(manifest[requested_key], "homogeneous")
                    other_key = "reference_ipm_method" if side == "candidate" else "ipm_method"
                    self.assertEqual(manifest[other_key], "mehrotra" if explicit_default else None)
                    for record in records:
                        enabled = record["requested_variant"] == summary[side + "_variant"]
                        actual = record["report"]["options"]["ipm"]
                        expected = "homogeneous" if enabled else "mehrotra"
                        self.assertEqual(actual["method"], expected)
                        self.assertEqual(":method=homogeneous" in record["requested_variant"], enabled)
                        self.assertNotIn(":method=mehrotra", record["requested_variant"])
                        self.assertEqual("--ipm-method" in record["command"], enabled or explicit_default)
                        configuration = manifest["variants"][record["requested_variant"]]
                        self.assertEqual(configuration["ipm_method"], expected if enabled or explicit_default else None)
                        self.assertEqual(actual["central_corrections"], 0 if enabled else 1)
                        self.assertEqual(actual["krylov_history_size"], 0 if enabled else 1)
                        for field, count in (("ipm_homogeneous_constant_solves", 3), ("ipm_homogeneous_direction_solves", 6)):
                            self.assertIs(type(record["report"][field]), int)
                            self.assertEqual(record["report"][field], count if enabled else 0)
                        self.assertEqual(record["report"]["ipm_homogeneous_tau"], 1.5 if enabled else None)
                        self.assertEqual(record["report"]["ipm_homogeneous_kappa"], 1e-8 if enabled else None)

    def test_explicit_ipm_method_rejects_missing_wrong_types_and_ignored_request(self):
        for side in ("candidate", "reference"):
            for index, reported in enumerate((None, False, 1, 1.0, "mehrotra")):
                with self.subTest(side=side, reported=reported):
                    self.output = self.root / f"method-report-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported is None:
                                del value["options"]["ipm"]["method"]
                            else:
                                value["options"]["ipm"]["method"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-method", "homogeneous"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_method_mismatch": 2})

    def test_homogeneous_work_rejects_invalid_types_ranges_and_inconsistent_nulls(self):
        missing = object()
        counts = ("ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves")
        scalars = ("ipm_homogeneous_tau", "ipm_homogeneous_kappa")
        cases = [(field, value) for field in counts for value in (missing, None, False, 1.0, -1, 9223372036854775808)]
        cases += [(field, value) for field in scalars for value in (missing, None, False, "1", 0, -1)]
        cases.append(("both_null", None))
        for side in ("candidate", "reference"):
            for index, (field, invalid) in enumerate(cases):
                with self.subTest(side=side, field=field, invalid=invalid):
                    self.output = self.root / f"hsd-work-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if field == "both_null":
                                value.update({name: None for name in scalars})
                            elif invalid is missing:
                                del value[field]
                            else:
                                value[field] = invalid
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-method", "homogeneous"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_homogeneous_work_mismatch": 2})

    def test_homogeneous_uninitialized_statistics_are_valid_without_work(self):
        def responder(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "ipm":
                value.update(ipm_homogeneous_constant_solves=0, ipm_homogeneous_direction_solves=0,
                             ipm_homogeneous_tau=None, ipm_homogeneous_kappa=None)
                process.stdout = json.dumps(value).encode()
            return process
        status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1,
                                                      extra_args=["--candidate-backend", "ipm", "--ipm-method", "homogeneous"])
        self.assertEqual(status, 0)
        self.assertTrue(all(record["quality_pass"] for record in records))

    def test_old_ipm_reports_remain_compatible_when_method_is_not_requested(self):
        def responder(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "ipm":
                del value["options"]["ipm"]["method"]
                for field in tuple(value):
                    if field.startswith("ipm_homogeneous_"):
                        del value[field]
                process.stdout = json.dumps(value).encode()
            return process
        status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1,
                                                      extra_args=["--candidate-backend", "ipm"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["candidate_variant"], "ipm")
        self.assertTrue(all(record["quality_pass"] for record in records))
        self.assertTrue(all("--ipm-method" not in record["command"] for record in records))

    def test_homogeneous_combinations_are_rejected_before_process_start(self):
        combinations = [["ipm-linear-accuracy", accuracy] for accuracy in COMPARISON.benchmark.IPM_LINEAR_ACCURACIES
                        if accuracy not in ("strict", "homogeneous_scaled", "homogeneous_repair")]
        combinations += [["ipm-central-corrections", "1"], ["ipm-direction-retries", "1"],
                         ["ipm-krylov-history-size", "1"], ["ipm-inexact-prefilter", "on"]]
        for flag, value in combinations:
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         prefix + "ipm-method", "homogeneous", prefix + flag, value]
                with self.subTest(side=side, flag=flag, value=value), self.assertRaises(SystemExit) as error, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=flags)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", "ipm", "--ipm-method", "homogeneous", "--" + flag, value,
                         "--output-dir", str(self.output)]
            with self.subTest(benchmark=True, flag=flag, value=value), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_homogeneous_scaled_accuracy_is_independent_on_both_sides(self):
        for side in ("candidate", "reference"):
            for other_method in ("mehrotra", "homogeneous"):
                with self.subTest(side=side, other_method=other_method):
                    self.output = self.root / f"scaled-{side}-{other_method}"
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "--reference-" if side == "candidate" else "--"
                    flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                             prefix + "ipm-method", "homogeneous", prefix + "ipm-linear-accuracy", "homogeneous_scaled",
                             prefix + "ipm-central-corrections", "0", prefix + "ipm-direction-retries", "0",
                             prefix + "ipm-krylov-history-size", "0", prefix + "ipm-inexact-prefilter", "off",
                             other + "ipm-method", other_method, other + "ipm-linear-accuracy", "strict"]
                    status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 0)
                    manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                    for record in records:
                        enabled = record["requested_variant"] == summary[side + "_variant"]
                        accuracy = "homogeneous_scaled" if enabled else "strict"
                        method = "homogeneous" if enabled else other_method
                        self.assertEqual(record["report"]["options"]["ipm"]["linear_accuracy"], accuracy)
                        self.assertEqual(record["report"]["options"]["ipm"]["method"], method)
                        self.assertEqual(record["command"][record["command"].index("--ipm-linear-accuracy") + 1], accuracy)
                        self.assertIn(":linear=" + accuracy + ":", record["requested_variant"])
                        self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_linear_accuracy"], accuracy)
                        for field, count in (("ipm_homogeneous_scaled_products", 10),
                                             ("ipm_homogeneous_scaled_refinement_steps", 4),
                                             ("ipm_homogeneous_scaled_inexact_returns", 2)):
                            self.assertIs(type(record["report"][field]), int)
                            self.assertEqual(record["report"][field], count if enabled else 0)

    def test_homogeneous_scaled_rejects_ignored_accuracy_and_invalid_statistics(self):
        missing = object()
        fields = ("ipm_homogeneous_scaled_products", "ipm_homogeneous_scaled_refinement_steps",
                  "ipm_homogeneous_scaled_inexact_returns")
        cases = [(field, value, "ipm_homogeneous_scaled_work_mismatch") for field in fields
                 for value in (missing, None, False, 1.0, "1", -1, 9223372036854775808)]
        cases += [(fields[0], 1, "ipm_homogeneous_scaled_work_mismatch"),
                  (fields[1], 9, "ipm_homogeneous_scaled_work_mismatch"),
                  (fields[2], 9, "ipm_homogeneous_scaled_work_mismatch")]
        cases += [("linear_accuracy", value, "ipm_linear_accuracy_mismatch") for value in (missing, None, False, 1, "strict")]
        for side in ("candidate", "reference"):
            for index, (field, invalid, failure) in enumerate(cases):
                with self.subTest(side=side, field=field, invalid=invalid):
                    self.output = self.root / f"scaled-report-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            target = value["options"]["ipm"] if field == "linear_accuracy" else value
                            if invalid is missing:
                                del target[field]
                            else:
                                target[field] = invalid
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-method", "homogeneous", prefix + "ipm-linear-accuracy", "homogeneous_scaled"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"], {failure: 2})

    def test_homogeneous_scaled_combinations_fail_before_running(self):
        combinations = [[], ["ipm-method", "mehrotra"]]
        combinations += [["ipm-method", "homogeneous", flag, value] for flag, value in (
            ("ipm-central-corrections", "1"), ("ipm-direction-retries", "1"),
            ("ipm-krylov-history-size", "1"), ("ipm-inexact-prefilter", "on"))]
        for combination in combinations:
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                other = "--reference-" if side == "candidate" else "--"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         prefix + "ipm-linear-accuracy", "homogeneous_scaled", other + "ipm-method", "homogeneous"]
                flags += [prefix + item if i % 2 == 0 else item for i, item in enumerate(combination)]
                with self.subTest(side=side, combination=combination), self.assertRaises(SystemExit) as error, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=flags)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())
            flags = ["--" + item if i % 2 == 0 else item for i, item in enumerate(combination)]
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", "ipm", "--ipm-linear-accuracy", "homogeneous_scaled", *flags,
                         "--output-dir", str(self.output)]
            with self.subTest(benchmark=True, combination=combination), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_homogeneous_repair_accuracy_and_counts_are_independent(self):
        for side in ("candidate", "reference"):
            for other_method in ("mehrotra", "homogeneous"):
                with self.subTest(side=side, other_method=other_method):
                    self.output = self.root / f"repair-{side}-{other_method}"
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "--reference-" if side == "candidate" else "--"
                    flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                             prefix + "ipm-method", "homogeneous", prefix + "ipm-linear-accuracy", "homogeneous_repair",
                             prefix + "ipm-central-corrections", "0", prefix + "ipm-direction-retries", "0",
                             prefix + "ipm-krylov-history-size", "0", prefix + "ipm-inexact-prefilter", "off",
                             other + "ipm-method", other_method, other + "ipm-linear-accuracy", "strict"]
                    status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 0)
                    manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                    for record in records:
                        enabled = record["requested_variant"] == summary[side + "_variant"]
                        accuracy = "homogeneous_repair" if enabled else "strict"
                        self.assertEqual(record["report"]["options"]["ipm"]["linear_accuracy"], accuracy)
                        self.assertEqual(record["report"]["options"]["ipm"]["method"], "homogeneous" if enabled else other_method)
                        self.assertEqual(record["command"][record["command"].index("--ipm-linear-accuracy") + 1], accuracy)
                        self.assertIn(":linear=" + accuracy + ":", record["requested_variant"])
                        self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_linear_accuracy"], accuracy)
                        for field, count in (("ipm_homogeneous_constant_repairs", 2),
                                             ("ipm_homogeneous_constant_repair_successes", 1)):
                            self.assertIs(type(record["report"][field]), int)
                            self.assertEqual(record["report"][field], count if enabled else 0)

    def test_homogeneous_repair_rejects_invalid_reports_on_both_sides(self):
        missing = object()
        fields = ("ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes")
        failure = "ipm_homogeneous_repair_work_mismatch"
        cases = [(field, value, failure) for field in fields
                 for value in (missing, None, False, 1.0, "1", -1, 9223372036854775808)]
        cases += [(fields[0], 6, failure), (fields[1], 3, failure), ("ipm_homogeneous_constant_solves", 1, failure)]
        cases += [("linear_accuracy", value, "ipm_linear_accuracy_mismatch")
                  for value in (missing, None, False, 1, "strict", "homogeneous_scaled")]
        for side in ("candidate", "reference"):
            for index, (field, invalid, failure) in enumerate(cases):
                with self.subTest(side=side, field=field, invalid=invalid):
                    self.output = self.root / f"repair-report-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            target = value["options"]["ipm"] if field == "linear_accuracy" else value
                            if invalid is missing:
                                del target[field]
                            else:
                                target[field] = invalid
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-method", "homogeneous", prefix + "ipm-linear-accuracy", "homogeneous_repair"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"], {failure: 2})

    def test_homogeneous_repair_combinations_fail_before_running(self):
        combinations = [[], ["ipm-method", "mehrotra"]]
        combinations += [["ipm-method", "homogeneous", flag, value] for flag, value in (
            ("ipm-central-corrections", "1"), ("ipm-direction-retries", "1"),
            ("ipm-krylov-history-size", "1"), ("ipm-inexact-prefilter", "on"))]
        for combination in combinations:
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                other = "--reference-" if side == "candidate" else "--"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         prefix + "ipm-linear-accuracy", "homogeneous_repair", other + "ipm-method", "homogeneous"]
                flags += [prefix + item if i % 2 == 0 else item for i, item in enumerate(combination)]
                with self.subTest(side=side, combination=combination), self.assertRaises(SystemExit) as error, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=flags)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())
            flags = ["--" + item if i % 2 == 0 else item for i, item in enumerate(combination)]
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", "ipm", "--ipm-linear-accuracy", "homogeneous_repair", *flags,
                         "--output-dir", str(self.output)]
            with self.subTest(benchmark=True, combination=combination), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_homogeneous_repair_zero_work_is_valid(self):
        value = report("ipm")
        self.assertTrue(COMPARISON.benchmark.valid_homogeneous_repair_work(value))
        value.update(ipm_homogeneous_constant_repairs=1, ipm_homogeneous_constant_repair_successes=0,
                     ipm_homogeneous_constant_solves=2)
        self.assertTrue(COMPARISON.benchmark.valid_homogeneous_repair_work(value))

    def test_homogeneous_strict_old_reports_need_no_scaled_statistics(self):
        def responder(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            for field in tuple(value):
                if field.startswith("ipm_homogeneous_scaled_") or field in (
                    "ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes"
                ):
                    del value[field]
            process.stdout = json.dumps(value).encode()
            return process
        status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--ipm-method", "homogeneous", "--ipm-linear-accuracy", "strict"])
        self.assertEqual(status, 0)
        self.assertTrue(all(record["quality_pass"] for record in records))

    def test_bounded_newton_accuracy_and_budgets_are_independent(self):
        for side in ("candidate", "reference"):
            for budget in (None, 0):
                with self.subTest(side=side, budget=budget):
                    self.output = self.root / f"bounded-newton-{side}-{budget}"
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "--reference-" if side == "candidate" else "--"
                    flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                             prefix + "ipm-linear-accuracy", "bounded_newton", prefix + "ipm-krylov-history-size", "0",
                             prefix + "ipm-central-corrections", "0", other + "ipm-linear-accuracy", "strict",
                             other + "ipm-krylov-history-size", "1", other + "ipm-central-corrections", "1"]
                    if budget is not None:
                        flags += [prefix + "ipm-direction-retries", str(budget)]
                    status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 0)
                    manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                    key = "ipm_linear_accuracy" if side == "candidate" else "reference_ipm_linear_accuracy"
                    self.assertEqual(manifest[key], "bounded_newton")
                    for record in records:
                        enabled = record["requested_variant"] == summary[side + "_variant"]
                        actual = record["report"]["options"]["ipm"]
                        accuracy = "bounded_newton" if enabled else "strict"
                        self.assertEqual(actual["linear_accuracy"], accuracy)
                        command = record["command"]
                        self.assertEqual(command[command.index("--ipm-linear-accuracy") + 1], accuracy)
                        self.assertIn(":linear=" + accuracy + ":", record["requested_variant"])
                        configuration = manifest["variants"][record["requested_variant"]]
                        self.assertEqual(configuration["ipm_linear_accuracy"], accuracy)
                        self.assertEqual(actual["central_corrections"], 0 if enabled else 1)
                        self.assertEqual(actual["krylov_history_size"], 0 if enabled else 1)
                        self.assertEqual(actual["direction_retries"], (budget or 0) if enabled else 0)
                        if enabled:
                            self.assertEqual(configuration["ipm_direction_retries"], budget)
                            self.assertEqual("--ipm-direction-retries" in command, budget is not None)
                            self.assertEqual(":retries=" in record["requested_variant"], bool(budget))
                        for field, count in (("ipm_full_newton_preconditioner_solves", 6), ("ipm_full_newton_products", 9)):
                            self.assertIs(type(record["report"][field]), int)
                            self.assertEqual(record["report"][field], count if enabled else 0)

    def test_bounded_newton_report_rejects_missing_wrong_types_and_ignored_mode(self):
        for side in ("candidate", "reference"):
            for index, reported in enumerate((None, False, 1, 1.0, "strict")):
                with self.subTest(side=side, reported=reported):
                    self.output = self.root / f"bounded-newton-report-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported is None:
                                del value["options"]["ipm"]["linear_accuracy"]
                            else:
                                value["options"]["ipm"]["linear_accuracy"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-linear-accuracy", "bounded_newton"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_linear_accuracy_mismatch": 2})

    def test_full_newton_accuracy_and_budgets_are_independent(self):
        for side in ("candidate", "reference"):
            for budget in (None, 0, 1, 8):
                with self.subTest(side=side, budget=budget):
                    self.output = self.root / f"full-newton-{side}-{budget}"
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "--reference-" if side == "candidate" else "--"
                    flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                             prefix + "ipm-linear-accuracy", "full_newton", prefix + "ipm-krylov-history-size", "0",
                             prefix + "ipm-central-corrections", "0", other + "ipm-linear-accuracy", "strict",
                             other + "ipm-krylov-history-size", "1", other + "ipm-central-corrections", "1"]
                    if budget is not None:
                        flags += [prefix + "ipm-direction-retries", str(budget)]
                    status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 0)
                    manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                    key = "ipm_linear_accuracy" if side == "candidate" else "reference_ipm_linear_accuracy"
                    self.assertEqual(manifest[key], "full_newton")
                    for record in records:
                        enabled = record["requested_variant"] == summary[side + "_variant"]
                        actual = record["report"]["options"]["ipm"]
                        accuracy = "full_newton" if enabled else "strict"
                        self.assertEqual(actual["linear_accuracy"], accuracy)
                        command = record["command"]
                        self.assertEqual(command[command.index("--ipm-linear-accuracy") + 1], accuracy)
                        self.assertIn(":linear=" + accuracy + ":", record["requested_variant"])
                        configuration = manifest["variants"][record["requested_variant"]]
                        self.assertEqual(configuration["ipm_linear_accuracy"], accuracy)
                        self.assertEqual(actual["central_corrections"], 0 if enabled else 1)
                        self.assertEqual(actual["krylov_history_size"], 0 if enabled else 1)
                        self.assertEqual(actual["direction_retries"], (budget or 0) if enabled else 0)
                        if enabled:
                            self.assertEqual(configuration["ipm_direction_retries"], budget)
                            self.assertEqual("--ipm-direction-retries" in command, budget is not None)
                            self.assertEqual(":retries=" in record["requested_variant"], bool(budget))
                        for field, count in (("ipm_full_newton_preconditioner_solves", 6), ("ipm_full_newton_products", 9)):
                            self.assertIs(type(record["report"][field]), int)
                            self.assertEqual(record["report"][field], count if enabled else 0)

    def test_full_newton_report_rejects_missing_wrong_types_and_ignored_mode(self):
        for side in ("candidate", "reference"):
            for index, reported in enumerate((None, False, 1, 1.0, "strict")):
                with self.subTest(side=side, reported=reported):
                    self.output = self.root / f"full-newton-report-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported is None:
                                del value["options"]["ipm"]["linear_accuracy"]
                            else:
                                value["options"]["ipm"]["linear_accuracy"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    prefix = "--" if side == "candidate" else "--reference-"
                    flags = ["--candidate-backend", "ipm"] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm"]
                    flags += [prefix + "ipm-linear-accuracy", "full_newton"]
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_linear_accuracy_mismatch": 2})

    def test_full_newton_corrections_and_wrong_backends_fail_before_running(self):
        cases = [("ipm", ["--ipm-central-corrections", "1"]), ("highs", []), ("pdlp", [])]
        for backend, extra in cases:
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", backend, "--output-dir", str(self.output), "--ipm-linear-accuracy", "full_newton"] + extra
            with self.subTest(tool="benchmark", backend=backend), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                flags = ["--" + side + "-backend", backend, prefix + "ipm-linear-accuracy", "full_newton"]
                if extra:
                    flags += [prefix + "ipm-central-corrections", "1"]
                with self.subTest(tool="compare", side=side, backend=backend), \
                        self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=flags)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())

    def test_bounded_newton_budget_combinations_fail_before_running(self):
        for flag in ("ipm-central-corrections", "ipm-direction-retries", "ipm-krylov-history-size"):
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", "ipm", "--output-dir", str(self.output),
                         "--ipm-linear-accuracy", "bounded_newton", "--" + flag, "1"]
            with self.subTest(tool="benchmark", flag=flag), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            process.assert_not_called()
            self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                flags = ["--" + side + "-backend", "ipm", prefix + "ipm-linear-accuracy", "bounded_newton",
                         prefix + flag, "1"]
                with self.subTest(tool="compare", flag=flag, side=side), \
                        self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=flags)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())

    def test_bounded_newton_work_report_rejects_invalid_counts(self):
        for field in ("ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns",
                      "ipm_full_newton_preconditioner_solves", "ipm_full_newton_products"):
            for index, bad in enumerate((None, True, -1, 1.0, "2")):
                self.output = self.root / f"bad-work-{field}-{index}"
                def responder(command, **kwargs):
                    process = self.fake_process(command, **kwargs)
                    value = json.loads(process.stdout)
                    if value["backend"] == "ipm":
                        if bad is None:
                            del value[field]
                        else:
                            value[field] = bad
                        process.stdout = json.dumps(value).encode()
                    return process
                status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1,
                    extra_args=["--candidate-backend", "ipm", "--ipm-linear-accuracy", "bounded_newton"])
                self.assertEqual(status, 1)
                self.assertEqual(summary["variants"][summary["candidate_variant"]]["quality_failure_counts"],
                                 {"ipm_bounded_newton_work_mismatch": 2})

    def test_linear_options_report_requires_matching_string(self):
        for field, requested, other in (("linear_accuracy", "backward", "strict"),
                                        ("linear_accuracy", "inexact", "backward"),
                                        ("linear_accuracy", "blockwise", "strict"),
                                        ("linear_accuracy", "blockwise_weighted", "blockwise"),
                                        ("linear_solver", "sparselu", "qdldl"),
                                        ("linear_solver", "normal", "sparselu"),
                                        ("linear_solver", "normal_cholmod", "normal_augmented"),
                                        ("linear_solver", "normal_cholmod_aat", "normal_cholmod")):
            for index, reported in enumerate((None, 1, True, other)):
                with self.subTest(field=field, reported=reported):
                    self.output = self.root / f"bad-{field}-{requested}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported is None:
                                del value["options"]["ipm"][field]
                            else:
                                value["options"]["ipm"][field] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1,
                        extra_args=["--candidate-backend", "ipm", "--ipm-" + field.replace("_", "-"), requested]
                        + (["--ipm-singleton-elimination", "on"] if requested in ("normal", "normal_cholmod", "normal_cholmod_aat") else []))
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["quality_pass_count"], 2)
                    self.assertEqual(summary["variants"][summary["candidate_variant"]]["quality_failure_counts"],
                                     {"ipm_" + field + "_mismatch": 2})

    def test_linear_solver_is_independent_with_combined_configuration(self):
        cases = [("sparselu", None), (None, "sparselu"), ("qdldl", "sparselu"), ("normal", None),
                 ("qdldl_extended", None), (None, "qdldl_extended"),
                 ("factorhighs", None), (None, "factorhighs"),
                 ("normal_cholmod", None), (None, "normal_cholmod"),
                 ("normal_cholmod_aat", None), (None, "normal_cholmod_aat"),
                 ("normal_cholmod_aat", "normal_cholmod"), ("normal_cholmod", "normal_cholmod_aat"),
                 ("normal_cholmod", "normal_augmented"), ("normal_augmented", "normal_cholmod"),
                 ("normal_augmented", "normal"), ("normal", "normal_augmented"), ("normal_multifrontal", "normal_augmented"), ("normal_augmented", "normal_multifrontal"), ("normal_extended", "normal_augmented"), ("normal_augmented", "normal_extended"), ("normal_refined", "normal_augmented"), ("normal_augmented", "normal_refined")]
        for index, (candidate, reference) in enumerate(cases):
            with self.subTest(candidate=candidate, reference=reference):
                self.output = self.root / f"linear-solver-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         "--ipm-singleton-elimination", "on", "--ipm-kkt-equilibration", "on"]
                if candidate is not None:
                    flags += ["--ipm-linear-solver", candidate]
                if reference is not None:
                    flags += ["--reference-ipm-linear-solver", reference]
                reference_singletons = "on" if reference in ("normal", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined", "normal_cholmod", "normal_cholmod_aat") else "off"
                if reference_singletons == "on":
                    flags += ["--reference-ipm-singleton-elimination", "on"]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertEqual(summary["candidate_variant"],
                    "ipm:singletons=on:dynamic=off:equilibrate=on:linear=strict:corrections=0:solver=" + (candidate or "qdldl") + ":prefilter=off:fixed=off")
                self.assertEqual(summary["reference_variant"],
                    "ipm:singletons=" + reference_singletons + ":dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=" + (reference or "qdldl") + ":prefilter=off:fixed=off")
                self.assertIsNone(summary["backends"])
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_linear_solver"], candidate)
                self.assertEqual(manifest["reference_ipm_linear_solver"], reference)
                for record in records:
                    requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                    self.assertEqual(record["report"]["options"]["ipm"]["linear_solver"], requested or "qdldl")
                    self.assertEqual("--ipm-linear-solver" in record["command"], requested is not None)
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_linear_solver"], requested)

    def test_cholmod_requires_its_own_singleton_switch_and_rejects_dynamic_before_running(self):
        cases = [("ipm", []), ("ipm", ["singleton-elimination", "off"]),
                 ("ipm", ["singleton-elimination", "on", "dynamic-regularization", "on"]),
                 ("highs", ["singleton-elimination", "on"]),
                 ("pdlp", ["singleton-elimination", "on"])]
        for linear in ("normal_cholmod", "normal_cholmod_aat"):
            for backend, switches in cases:
                flags = ["--ipm-linear-solver", linear]
                for index in range(0, len(switches), 2):
                    flags += ["--ipm-" + switches[index], switches[index + 1]]
                with self.subTest(linear=linear, tool="benchmark", backend=backend, switches=switches):
                    arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                                 "--backend", backend, "--output-dir", str(self.output), *flags]
                    with mock.patch.object(sys, "argv", arguments), \
                            mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                            self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                        COMPARISON.benchmark.main()
                    self.assertEqual(error.exception.code, 2)
                    process.assert_not_called()
                    self.assertFalse(self.output.exists())
                for side in ("candidate", "reference"):
                    with self.subTest(linear=linear, tool="compare", side=side, backend=backend, switches=switches):
                        # 另一侧开启单行消元也不能替代本侧必需的配置。
                        prefix = "--" if side == "candidate" else "--reference-"
                        other = "reference" if side == "candidate" else "candidate"
                        other_prefix = "--reference-" if side == "candidate" else "--"
                        side_flags = [prefix + flag[2:] if flag.startswith("--") else flag for flag in flags]
                        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                            self.run_comparison(default_pdlp=False, extra_args=[
                                "--" + side + "-backend", backend, "--" + other + "-backend", "ipm",
                                other_prefix + "ipm-singleton-elimination", "on", *side_flags])
                        self.assertEqual(error.exception.code, 2)
                        self.assertFalse(self.commands)
                        self.assertFalse(self.output.exists())

    def test_inexact_mode_preserves_counts_and_independent_solver_combinations(self):
        for index, (candidate_accuracy, reference_accuracy) in enumerate((("inexact", None), ("strict", "inexact"),
                                                                         ("blockwise", "strict"), ("strict", "blockwise"),
                                                                         ("blockwise_weighted", "blockwise"), ("blockwise", "blockwise_weighted"),
                                                                         ("blockwise_forcing", "strict"), ("strict", "blockwise_forcing"),
                                                                         ("blockwise_forcing_weighted", "blockwise_forcing"), ("blockwise_forcing", "blockwise_forcing_weighted"),
                                                                         ("adaptive_regularization", "strict"), ("strict", "adaptive_regularization"),
                                                                         ("primal_dual_regularization", "strict"), ("strict", "primal_dual_regularization"),
                                                                         ("curvature_regularization", "strict"), ("strict", "curvature_regularization"))):
            with self.subTest(candidate=candidate_accuracy, reference=reference_accuracy):
                self.output = self.root / f"inexact-combinations-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         "--ipm-linear-accuracy", candidate_accuracy, "--ipm-linear-solver", "sparselu",
                         "--ipm-singleton-elimination", "on", "--ipm-kkt-equilibration", "on"]
                if reference_accuracy is not None:
                    flags += ["--reference-ipm-linear-accuracy", reference_accuracy]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertEqual(summary["candidate_variant"],
                    f"ipm:singletons=on:dynamic=off:equilibrate=on:linear={candidate_accuracy}:corrections=0:solver=sparselu:prefilter=off:fixed=off")
                self.assertEqual(summary["reference_variant"],
                    f"ipm:singletons=off:dynamic=off:equilibrate=off:linear={reference_accuracy or 'strict'}:corrections=0:solver=qdldl:prefilter=off:fixed=off")
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_linear_accuracy"], candidate_accuracy)
                self.assertEqual(manifest["reference_ipm_linear_accuracy"], reference_accuracy)
                for record in records:
                    is_candidate = record["requested_variant"] == summary["candidate_variant"]
                    requested = candidate_accuracy if is_candidate else reference_accuracy
                    self.assertEqual(record["report"]["options"]["ipm"]["linear_accuracy"], requested or "strict")
                    self.assertEqual(record["report"]["options"]["ipm"]["linear_solver"], "sparselu" if is_candidate else "qdldl")
                    self.assertEqual("--ipm-linear-accuracy" in record["command"], requested is not None)
                    self.assertEqual(record["report"]["ipm_inexact_acceptances"], 5 if requested == "inexact" else 0)
                    self.assertEqual(record["report"]["ipm_backward_error_acceptances"], 0)
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_linear_accuracy"], requested)

    def test_objective_scaling_has_independent_configuration_and_default(self):
        for index, (candidate, reference) in enumerate((("off", None), (None, "off"), ("off", "on"), ("on", "off"))):
            self.output = self.root / f"objective-scaling-{index}"
            flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
            if candidate is not None:
                flags += ["--ipm-objective-scaling", candidate]
            if reference is not None:
                flags += ["--reference-ipm-objective-scaling", reference]
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
            self.assertEqual(status, 0)
            manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["ipm_objective_scaling"], candidate)
            self.assertEqual(manifest["reference_ipm_objective_scaling"], reference)
            for record in records:
                variant = record["requested_variant"]
                requested = candidate if variant == summary["candidate_variant"] else reference
                self.assertIs(record["report"]["options"]["ipm"]["objective_scaling"], requested != "off")
                self.assertEqual("--ipm-objective-scaling" in record["command"], requested is not None)
                self.assertEqual(variant.endswith(":objective_scaling=off"), requested == "off")
                self.assertEqual(manifest["variants"][variant]["ipm_objective_scaling"], requested)

    def test_objective_scaling_report_requires_matching_boolean_on_both_sides(self):
        responses = [("on", "missing"), ("on", None), ("on", 1), ("on", "on"), ("on", False),
                     ("off", "missing"), ("off", None), ("off", 0), ("off", "off"), ("off", True)]
        for side in ("candidate", "reference"):
            for index, (requested, reported) in enumerate(responses):
                with self.subTest(side=side, requested=requested, reported=reported):
                    self.output = self.root / f"bad-objective-scaling-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported == "missing":
                                del value["options"]["ipm"]["objective_scaling"]
                            else:
                                value["options"]["ipm"]["objective_scaling"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    flags = ["--candidate-backend", "ipm", "--ipm-objective-scaling", requested] if side == "candidate" else [
                        "--candidate-backend", "highs", "--reference-backend", "ipm", "--reference-ipm-objective-scaling", requested]
                    status, records, summary = self.run_comparison(responder, repeats=1, default_pdlp=False, extra_args=flags)
                    self.assertEqual(status, 1)
                    variant = summary[side + "_variant"]
                    self.assertEqual(summary["variants"][variant]["quality_failure_counts"], {"ipm_objective_scaling_mismatch": 2})
                    self.assertEqual(summary["quality_pass_count"], 2)

    def test_fixed_variable_elimination_has_independent_configuration_and_counts(self):
        for index, (candidate, reference) in enumerate((("on", None), (None, "on"), ("off", "on"))):
            self.output = self.root / f"fixed-variables-{index}"
            flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
            if candidate is not None:
                flags += ["--ipm-fixed-variable-elimination", candidate]
            if reference is not None:
                flags += ["--reference-ipm-fixed-variable-elimination", reference]
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
            self.assertEqual(status, 0)
            self.assertTrue(summary["candidate_variant"].endswith(":fixed=" + (candidate or "off")))
            self.assertTrue(summary["reference_variant"].endswith(":fixed=" + (reference or "off")))
            manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["ipm_fixed_variable_elimination"], candidate)
            self.assertEqual(manifest["reference_ipm_fixed_variable_elimination"], reference)
            for record in records:
                requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                self.assertIs(record["report"]["options"]["ipm"]["eliminate_fixed_variables"], requested == "on")
                self.assertEqual("--ipm-fixed-variable-elimination" in record["command"], requested is not None)
                self.assertEqual(record["report"]["ipm_eliminated_fixed_variables"], 1 if requested == "on" else 0)
                self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_fixed_variable_elimination"], requested)

    def test_singleton_presolve_has_independent_configuration_and_counts(self):
        prefix = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
        for index, (candidate, reference) in enumerate((("on", None), (None, "on"), ("off", "on"), ("on", "off"))):
            self.output = self.root / f"singleton-presolve-{index}"
            flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
            if candidate is not None:
                flags += ["--ipm-singleton-presolve", candidate]
            if reference is not None:
                flags += ["--reference-ipm-singleton-presolve", reference]
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
            self.assertEqual(status, 0)
            self.assertEqual(summary["candidate_variant"], prefix + (":presolve_singletons=on" if candidate == "on" else ""))
            self.assertEqual(summary["reference_variant"], prefix + (":presolve_singletons=on" if reference == "on" else ""))
            manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["ipm_singleton_presolve"], candidate)
            self.assertEqual(manifest["reference_ipm_singleton_presolve"], reference)
            for record in records:
                requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                self.assertIs(record["report"]["options"]["ipm"]["presolve_singleton_rows"], requested == "on")
                self.assertEqual("--ipm-singleton-presolve" in record["command"], requested is not None)
                if requested is not None:
                    self.assertEqual(record["command"][record["command"].index("--ipm-singleton-presolve") + 1], requested)
                self.assertEqual(record["report"]["ipm_presolved_singleton_rows"], 2 if requested == "on" else 0)
                self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_singleton_presolve"], requested)
        for value in ("off", "on"):
            self.output = self.root / ("singleton-presolve-legacy-" + value)
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
                "--candidate-backend", "ipm", "--ipm-singleton-presolve", value])
            self.assertEqual(status, 0)
            self.assertEqual(summary["candidate_variant"], "ipm" + (":presolve_singletons=on" if value == "on" else ""))
        self.output = self.root / "singleton-presolve-retries-label"
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--ipm-direction-retries", "8", "--ipm-singleton-presolve", "on"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["candidate_variant"], "ipm:retries=8:presolve_singletons=on")

    def test_reference_singleton_presolve_report_requires_matching_boolean(self):
        responses = [("on", "missing"), ("on", None), ("on", 1), ("on", "on"), ("on", False),
                     ("off", "missing"), ("off", None), ("off", 0), ("off", "off"), ("off", True)]
        for index, (requested, reported) in enumerate(responses):
            with self.subTest(requested=requested, reported=reported):
                self.output = self.root / f"bad-reference-singleton-presolve-{index}"
                def responder(command, **kwargs):
                    process = self.fake_process(command, **kwargs)
                    value = json.loads(process.stdout)
                    if value["backend"] == "ipm":
                        if reported == "missing":
                            del value["options"]["ipm"]["presolve_singleton_rows"]
                        else:
                            value["options"]["ipm"]["presolve_singleton_rows"] = reported
                        process.stdout = json.dumps(value).encode()
                    return process
                status, records, summary = self.run_comparison(responder, repeats=1, default_pdlp=False,
                    extra_args=["--candidate-backend", "highs", "--reference-backend", "ipm",
                                "--reference-ipm-singleton-presolve", requested])
                self.assertEqual(status, 1)
                self.assertEqual(summary["quality_pass_count"], 2)
                self.assertEqual(summary["variants"][summary["reference_variant"]]["quality_failure_counts"],
                                 {"ipm_singleton_presolve_mismatch": 2})
                self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                    if record["requested_backend"] == "ipm"))

    def test_singleton_presolve_rejects_invalid_and_non_ipm_arguments(self):
        cases = [(backend, value) for backend in ("highs", "pdlp") for value in ("on", "off")]
        cases += [("ipm", "true")]
        for backend, value in cases:
            with self.subTest(tool="benchmark", backend=backend, value=value):
                arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                             "--backend", backend, "--output-dir", str(self.output), "--ipm-singleton-presolve", value]
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                        self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                    COMPARISON.benchmark.main()
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()
                self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                with self.subTest(tool="compare", side=side, backend=backend, value=value):
                    flag = "--ipm-singleton-presolve" if side == "candidate" else "--reference-ipm-singleton-presolve"
                    with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                        self.run_comparison(default_pdlp=False, extra_args=["--" + side + "-backend", backend, flag, value])
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.commands)
                    self.assertFalse(self.output.exists())

    def test_presolve_fixed_elimination_has_independent_configuration_and_counts(self):
        prefix = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off:presolve_singletons=on"
        for index, (candidate, reference) in enumerate((("on", None), (None, "on"), ("off", "on"), ("on", "off"))):
            self.output = self.root / f"presolve-fixed-{index}"
            flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                     "--ipm-singleton-presolve", "on", "--reference-ipm-singleton-presolve", "on"]
            if candidate is not None:
                flags += ["--ipm-presolve-fixed-elimination", candidate]
            if reference is not None:
                flags += ["--reference-ipm-presolve-fixed-elimination", reference]
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
            self.assertEqual(status, 0)
            self.assertEqual(summary["candidate_variant"], prefix + (":presolve_fixed=on" if candidate == "on" else ""))
            self.assertEqual(summary["reference_variant"], prefix + (":presolve_fixed=on" if reference == "on" else ""))
            manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["ipm_presolve_fixed_elimination"], candidate)
            self.assertEqual(manifest["reference_ipm_presolve_fixed_elimination"], reference)
            for record in records:
                requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                self.assertIs(record["report"]["options"]["ipm"]["eliminate_presolved_fixed_variables"], requested == "on")
                self.assertEqual("--ipm-presolve-fixed-elimination" in record["command"], requested is not None)
                if requested is not None:
                    self.assertEqual(record["command"][record["command"].index("--ipm-presolve-fixed-elimination") + 1], requested)
                self.assertEqual(record["report"]["ipm_presolved_fixed_columns"], 1 if requested == "on" else 0)
                self.assertLessEqual(record["report"]["ipm_presolved_fixed_columns"], record["report"]["ipm_eliminated_fixed_variables"])
                self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_presolve_fixed_elimination"], requested)
        self.output = self.root / "presolve-fixed-off-legacy"
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--ipm-presolve-fixed-elimination", "off"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["candidate_variant"], "ipm")

    def test_presolve_fixed_report_requires_matching_boolean_on_both_sides(self):
        responses = [("on", "missing"), ("on", None), ("on", 1), ("on", "on"), ("on", False),
                     ("off", "missing"), ("off", None), ("off", 0), ("off", "off"), ("off", True)]
        for side in ("candidate", "reference"):
            for index, (requested, reported) in enumerate(responses):
                with self.subTest(side=side, requested=requested, reported=reported):
                    self.output = self.root / f"bad-{side}-presolve-fixed-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported == "missing":
                                del value["options"]["ipm"]["eliminate_presolved_fixed_variables"]
                            else:
                                value["options"]["ipm"]["eliminate_presolved_fixed_variables"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    flags = (["--candidate-backend", "ipm", "--ipm-singleton-presolve", "on",
                              "--ipm-presolve-fixed-elimination", requested] if side == "candidate" else
                             ["--candidate-backend", "highs", "--reference-backend", "ipm",
                              "--reference-ipm-singleton-presolve", "on", "--reference-ipm-presolve-fixed-elimination", requested])
                    status, records, summary = self.run_comparison(responder, repeats=1, default_pdlp=False, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["quality_pass_count"], 2)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_presolve_fixed_elimination_mismatch": 2})
                    self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                        if record["requested_backend"] == "ipm"))

    def test_presolve_fixed_arguments_require_own_side_presolve_and_ipm(self):
        cases = [(backend, value, None) for backend in ("highs", "pdlp") for value in ("on", "off")]
        cases += [("ipm", "true", "on"), ("ipm", "on", None), ("ipm", "on", "off")]
        for backend, value, presolve in cases:
            with self.subTest(tool="benchmark", backend=backend, value=value, presolve=presolve):
                arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                             "--backend", backend, "--output-dir", str(self.output), "--ipm-presolve-fixed-elimination", value]
                if presolve is not None:
                    arguments += ["--ipm-singleton-presolve", presolve]
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                        self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                    COMPARISON.benchmark.main()
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()
                self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                with self.subTest(tool="compare", side=side, backend=backend, value=value, presolve=presolve):
                    prefix = "--" if side == "candidate" else "--reference-"
                    other = "reference" if side == "candidate" else "candidate"
                    other_prefix = "--reference-" if side == "candidate" else "--"
                    flags = ["--" + side + "-backend", backend, prefix + "ipm-presolve-fixed-elimination", value,
                             "--" + other + "-backend", "ipm", other_prefix + "ipm-singleton-presolve", "on"]
                    if presolve is not None:
                        flags += [prefix + "ipm-singleton-presolve", presolve]
                    with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                        self.run_comparison(default_pdlp=False, extra_args=flags)
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.commands)
                    self.assertFalse(self.output.exists())

    def test_inexact_prefilter_is_independent_and_preserves_counts(self):
        for index, (candidate, reference) in enumerate((("on", None), (None, "on"), ("off", "on"))):
            with self.subTest(candidate=candidate, reference=reference):
                self.output = self.root / f"prefilter-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         "--ipm-linear-accuracy", "inexact", "--reference-ipm-linear-accuracy", "inexact",
                         "--ipm-linear-solver", "sparselu", "--ipm-singleton-elimination", "on"]
                if candidate is not None:
                    flags += ["--ipm-inexact-prefilter", candidate]
                if reference is not None:
                    flags += ["--reference-ipm-inexact-prefilter", reference]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertEqual(summary["candidate_variant"],
                    "ipm:singletons=on:dynamic=off:equilibrate=off:linear=inexact:corrections=0:solver=sparselu:prefilter=" + (candidate or "off") + ":fixed=off")
                self.assertEqual(summary["reference_variant"],
                    "ipm:singletons=off:dynamic=off:equilibrate=off:linear=inexact:corrections=0:solver=qdldl:prefilter=" + (reference or "off") + ":fixed=off")
                self.assertIsNone(summary["backends"])
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_inexact_prefilter"], candidate)
                self.assertEqual(manifest["reference_ipm_inexact_prefilter"], reference)
                for record in records:
                    requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                    self.assertIs(record["report"]["options"]["ipm"]["inexact_prefilter"], requested == "on")
                    self.assertEqual("--ipm-inexact-prefilter" in record["command"], requested is not None)
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_inexact_prefilter"], requested)
                    self.assertEqual(record["report"]["ipm_inexact_prefilter_rejections"], 7 if requested == "on" else 0)
                    self.assertEqual(record["report"]["ipm_inexact_acceptances"], 5)

    def test_central_corrections_are_independent_and_preserve_counts(self):
        for index, (candidate, reference) in enumerate(((1, None), (None, 1), (0, 1))):
            with self.subTest(candidate=candidate, reference=reference):
                self.output = self.root / f"corrections-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
                if candidate is not None:
                    flags += ["--ipm-central-corrections", str(candidate)]
                if reference is not None:
                    flags += ["--reference-ipm-central-corrections", str(reference)]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                prefix = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections="
                self.assertEqual(summary["candidate_variant"], prefix + str(candidate or 0) + ":solver=qdldl:prefilter=off:fixed=off")
                self.assertEqual(summary["reference_variant"], prefix + str(reference or 0) + ":solver=qdldl:prefilter=off:fixed=off")
                self.assertIsNone(summary["backends"])
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_central_corrections"], candidate)
                self.assertEqual(manifest["reference_ipm_central_corrections"], reference)
                for record in records:
                    requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                    actual = record["report"]["options"]["ipm"]["central_corrections"]
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, requested or 0)
                    self.assertEqual("--ipm-central-corrections" in record["command"], requested is not None)
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_central_corrections"], requested)
                    self.assertEqual(record["report"]["ipm_central_correction_attempts"], 5 if actual else 0)
                    self.assertEqual(record["report"]["ipm_central_correction_acceptances"], 2 if actual else 0)

    def test_direction_retries_are_independent_and_preserve_zero_labels(self):
        prefix = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
        for index, (candidate, reference) in enumerate(((8, None), (None, 3), (0, 1), (2, 0), (2, 8))):
            with self.subTest(candidate=candidate, reference=reference):
                self.output = self.root / f"direction-retries-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
                if candidate is not None:
                    flags += ["--ipm-direction-retries", str(candidate)]
                if reference is not None:
                    flags += ["--reference-ipm-direction-retries", str(reference)]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertEqual(summary["candidate_variant"], prefix + (f":retries={candidate}" if candidate else ""))
                self.assertEqual(summary["reference_variant"], prefix + (f":retries={reference}" if reference else ""))
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_direction_retries"], candidate)
                self.assertEqual(manifest["reference_ipm_direction_retries"], reference)
                for record in records:
                    requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                    actual = record["report"]["options"]["ipm"]["direction_retries"]
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, requested or 0)
                    self.assertEqual("--ipm-direction-retries" in record["command"], requested is not None)
                    if requested is not None:
                        self.assertEqual(record["command"][record["command"].index("--ipm-direction-retries") + 1], str(requested))
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_direction_retries"], requested)
        self.output = self.root / "direction-retries-zero-legacy"
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--ipm-direction-retries", "0"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["candidate_variant"], "ipm")
        self.assertTrue(all("--ipm-direction-retries" in record["command"] for record in records
                            if record["requested_backend"] == "ipm"))

    def test_krylov_history_size_are_independent_and_preserve_zero_labels(self):
        prefix = "ipm:singletons=off:dynamic=off:equilibrate=off:linear=strict:corrections=0:solver=qdldl:prefilter=off:fixed=off"
        for index, (candidate, reference) in enumerate(((8, None), (None, 3), (0, 1), (2, 0), (2, 8))):
            with self.subTest(candidate=candidate, reference=reference):
                self.output = self.root / f"krylov-history-size-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm"]
                if candidate is not None:
                    flags += ["--ipm-krylov-history-size", str(candidate)]
                if reference is not None:
                    flags += ["--reference-ipm-krylov-history-size", str(reference)]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                self.assertEqual(summary["candidate_variant"], prefix + (f":krylov_history={candidate}" if candidate else ""))
                self.assertEqual(summary["reference_variant"], prefix + (f":krylov_history={reference}" if reference else ""))
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_krylov_history_size"], candidate)
                self.assertEqual(manifest["reference_ipm_krylov_history_size"], reference)
                for record in records:
                    requested = candidate if record["requested_variant"] == summary["candidate_variant"] else reference
                    actual = record["report"]["options"]["ipm"]["krylov_history_size"]
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, requested or 0)
                    self.assertEqual("--ipm-krylov-history-size" in record["command"], requested is not None)
                    if requested is not None:
                        self.assertEqual(record["command"][record["command"].index("--ipm-krylov-history-size") + 1], str(requested))
                    self.assertEqual(manifest["variants"][record["requested_variant"]]["ipm_krylov_history_size"], requested)
        self.output = self.root / "krylov-history-size-zero-legacy"
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--ipm-krylov-history-size", "0"])
        self.assertEqual(status, 0)
        self.assertEqual(summary["candidate_variant"], "ipm")
        self.assertTrue(all("--ipm-krylov-history-size" in record["command"] for record in records
                            if record["requested_backend"] == "ipm"))


    def test_krylov_history_size_report_requires_matching_integer_on_both_sides(self):
        cases = [(8, "missing"), (8, None), (8, True), (8, 8.0), (8, "8"), (8, 0), (8, 9),
                 (0, "missing"), (0, False), (0, 0.0), (0, "0"), (0, 1), (0, -1)]
        for side in ("candidate", "reference"):
            for index, (requested, reported) in enumerate(cases):
                with self.subTest(side=side, requested=requested, reported=reported):
                    self.output = self.root / f"bad-krylov-history-size-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported == "missing":
                                del value["options"]["ipm"]["krylov_history_size"]
                            else:
                                value["options"]["ipm"]["krylov_history_size"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    flags = (["--candidate-backend", "ipm", "--ipm-krylov-history-size", str(requested)]
                             if side == "candidate" else ["--candidate-backend", "highs", "--reference-backend", "ipm",
                                                        "--reference-ipm-krylov-history-size", str(requested)])
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["quality_pass_count"], 2)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_krylov_history_size_mismatch": 2})
                    self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                        if record["requested_backend"] == "ipm"))


    def test_krylov_history_size_invalid_arguments_fail_before_running(self):
        cases = [(backend, value) for backend in ("highs", "pdlp") for value in ("0", "8")]
        cases += [("ipm", value) for value in ("-1", "9", "true", "1.0")]
        for backend, value in cases:
            with self.subTest(tool="benchmark", backend=backend, value=value):
                arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                             "--backend", backend, "--output-dir", str(self.output), "--ipm-krylov-history-size", value]
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                        self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                    COMPARISON.benchmark.main()
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()
                self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                with self.subTest(tool="compare", side=side, backend=backend, value=value):
                    flag = "--ipm-krylov-history-size" if side == "candidate" else "--reference-ipm-krylov-history-size"
                    with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                        self.run_comparison(default_pdlp=False, extra_args=["--" + side + "-backend", backend, flag, value])
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.commands)
                    self.assertFalse(self.output.exists())
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            self.run_comparison(default_pdlp=False, extra_args=["--candidate-backend", "ipm", "--reference-backend", "ipm",
                                                               "--ipm-krylov-history-size", "0"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.commands)
        self.assertFalse(self.output.exists())


    def test_blockwise_direction_retries_keep_independent_accuracy_and_budget(self):
        cases = [("blockwise", 8, "blockwise_weighted", 0),
                 ("blockwise_weighted", 8, "blockwise", 1),
                 ("blockwise", 0, "blockwise_weighted", 8),
                 ("strict", None, "blockwise", 8)]
        for index, (candidate_accuracy, candidate_retries, reference_accuracy, reference_retries) in enumerate(cases):
            with self.subTest(candidate=(candidate_accuracy, candidate_retries), reference=(reference_accuracy, reference_retries)):
                self.output = self.root / f"blockwise-direction-retries-{index}"
                flags = ["--candidate-backend", "ipm", "--reference-backend", "ipm",
                         "--ipm-linear-accuracy", candidate_accuracy,
                         "--reference-ipm-linear-accuracy", reference_accuracy,
                         "--reference-ipm-direction-retries", str(reference_retries)]
                if candidate_retries is not None:
                    flags += ["--ipm-direction-retries", str(candidate_retries)]
                status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=flags)
                self.assertEqual(status, 0)
                manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["ipm_linear_accuracy"], candidate_accuracy)
                self.assertEqual(manifest["reference_ipm_linear_accuracy"], reference_accuracy)
                self.assertEqual(manifest["ipm_direction_retries"], candidate_retries)
                self.assertEqual(manifest["reference_ipm_direction_retries"], reference_retries)
                for record in records:
                    is_candidate = record["requested_variant"] == summary["candidate_variant"]
                    accuracy = candidate_accuracy if is_candidate else reference_accuracy
                    retries = candidate_retries if is_candidate else reference_retries
                    expected_label = (f"ipm:singletons=off:dynamic=off:equilibrate=off:linear={accuracy}"
                                      ":corrections=0:solver=qdldl:prefilter=off:fixed=off")
                    if retries:
                        expected_label += f":retries={retries}"
                    self.assertEqual(record["requested_variant"], expected_label)
                    actual = record["report"]["options"]["ipm"]
                    self.assertEqual(actual["linear_accuracy"], accuracy)
                    self.assertIs(type(actual["direction_retries"]), int)
                    self.assertEqual(actual["direction_retries"], retries or 0)
                    self.assertEqual(record["command"][record["command"].index("--ipm-linear-accuracy") + 1], accuracy)
                    self.assertEqual("--ipm-direction-retries" in record["command"], retries is not None)
                    if retries is not None:
                        self.assertEqual(record["command"][record["command"].index("--ipm-direction-retries") + 1], str(retries))
                    configuration = manifest["variants"][record["requested_variant"]]
                    self.assertEqual(configuration["ipm_linear_accuracy"], accuracy)
                    self.assertEqual(configuration["ipm_direction_retries"], retries)

    def test_direction_retries_report_requires_matching_integer_on_both_sides(self):
        cases = [(8, "missing"), (8, None), (8, True), (8, 8.0), (8, "8"), (8, 0), (8, 9),
                 (0, "missing"), (0, False), (0, 0.0), (0, "0"), (0, 1), (0, -1)]
        for side in ("candidate", "reference"):
            for index, (requested, reported) in enumerate(cases):
                with self.subTest(side=side, requested=requested, reported=reported):
                    self.output = self.root / f"bad-direction-retries-{side}-{index}"
                    def responder(command, **kwargs):
                        process = self.fake_process(command, **kwargs)
                        value = json.loads(process.stdout)
                        if value["backend"] == "ipm":
                            if reported == "missing":
                                del value["options"]["ipm"]["direction_retries"]
                            else:
                                value["options"]["ipm"]["direction_retries"] = reported
                            process.stdout = json.dumps(value).encode()
                        return process
                    flags = (["--candidate-backend", "ipm", "--ipm-direction-retries", str(requested)]
                             if side == "candidate" else ["--candidate-backend", "highs", "--reference-backend", "ipm",
                                                        "--reference-ipm-direction-retries", str(requested)])
                    status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1, extra_args=flags)
                    self.assertEqual(status, 1)
                    self.assertEqual(summary["quality_pass_count"], 2)
                    self.assertEqual(summary["variants"][summary[side + "_variant"]]["quality_failure_counts"],
                                     {"ipm_direction_retries_mismatch": 2})
                    self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                        if record["requested_backend"] == "ipm"))

    def test_direction_retries_invalid_arguments_fail_before_running(self):
        cases = [(backend, value) for backend in ("highs", "pdlp") for value in ("0", "8")]
        cases += [("ipm", value) for value in ("-1", "9", "true", "1.0")]
        for backend, value in cases:
            with self.subTest(tool="benchmark", backend=backend, value=value):
                arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                             "--backend", backend, "--output-dir", str(self.output), "--ipm-direction-retries", value]
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                        self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                    COMPARISON.benchmark.main()
                self.assertEqual(error.exception.code, 2)
                process.assert_not_called()
                self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                with self.subTest(tool="compare", side=side, backend=backend, value=value):
                    flag = "--ipm-direction-retries" if side == "candidate" else "--reference-ipm-direction-retries"
                    with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                        self.run_comparison(default_pdlp=False, extra_args=["--" + side + "-backend", backend, flag, value])
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(self.commands)
                    self.assertFalse(self.output.exists())
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            self.run_comparison(default_pdlp=False, extra_args=["--candidate-backend", "ipm", "--reference-backend", "ipm",
                                                               "--ipm-direction-retries", "0"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.commands)
        self.assertFalse(self.output.exists())

    def test_krylov_history_accuracy_and_zero_budget_remain_independent(self):
        for index, accuracy in enumerate(COMPARISON.benchmark.IPM_LINEAR_ACCURACIES):
            self.output = self.root / f"krylov-zero-accuracy-{index}"
            method = ["--ipm-method", "homogeneous"] if accuracy in ("homogeneous_scaled", "homogeneous_repair") else []
            status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
                "--candidate-backend", "ipm", "--reference-backend", "ipm",
                "--ipm-linear-accuracy", accuracy, "--ipm-krylov-history-size", "0",
                "--reference-ipm-linear-accuracy", "strict", "--reference-ipm-krylov-history-size", "8", *method])
            self.assertEqual(status, 0)
            for record in records:
                candidate = record["requested_variant"] == summary["candidate_variant"]
                options = record["report"]["options"]["ipm"]
                self.assertEqual(options["linear_accuracy"], accuracy if candidate else "strict")
                self.assertEqual(options["krylov_history_size"], 0 if candidate else 8)
                self.assertEqual(record["report"]["ipm_krylov_recycle_peak_dimension"], 0 if candidate else 8)
        self.output = self.root / "krylov-blockwise-with-old-suffixes"
        status, records, summary = self.run_comparison(default_pdlp=False, repeats=1, extra_args=[
            "--candidate-backend", "ipm", "--reference-backend", "ipm",
            "--ipm-linear-accuracy", "blockwise", "--ipm-krylov-history-size", "1",
            "--ipm-direction-retries", "8", "--ipm-singleton-presolve", "on",
            "--ipm-presolve-fixed-elimination", "on", "--reference-ipm-krylov-history-size", "8"])
        self.assertEqual(status, 0)
        self.assertTrue(summary["candidate_variant"].endswith(":retries=8:presolve_singletons=on:presolve_fixed=on:krylov_history=1"))
        self.assertTrue(summary["reference_variant"].endswith(":krylov_history=8"))
        for record in records:
            actual = record["report"]["options"]["ipm"]["krylov_history_size"]
            for field in ("ipm_krylov_recycle_projections", "ipm_krylov_recycle_acceptances",
                          "ipm_krylov_recycle_vectors", "ipm_krylov_recycle_peak_dimension"):
                self.assertIs(type(record["report"][field]), int)
            self.assertLessEqual(record["report"]["ipm_krylov_recycle_peak_dimension"], actual)
            self.assertLessEqual(record["report"]["ipm_krylov_recycle_acceptances"], record["report"]["ipm_krylov_recycle_projections"])

    def test_benchmark_krylov_default_keeps_request_unspecified(self):
        arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                     "--backend", "ipm", "--output-dir", str(self.output)]
        with mock.patch.object(sys, "argv", arguments), \
                mock.patch.object(COMPARISON.benchmark.subprocess, "run", side_effect=self.fake_process), \
                contextlib.redirect_stdout(io.StringIO()):
            status = COMPARISON.benchmark.main()
        self.assertEqual(status, 0)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        record = json.loads((self.output / "runs.jsonl").read_text(encoding="utf-8"))
        self.assertIsNone(manifest["ipm_requested_options"]["krylov_history_size"])
        self.assertIs(type(record["report"]["options"]["ipm"]["krylov_history_size"]), int)
        self.assertEqual(record["report"]["options"]["ipm"]["krylov_history_size"], 0)
        self.assertNotIn("--ipm-krylov-history-size", record["command"])

    def test_positive_krylov_history_rejects_unsupported_accuracy_before_running(self):
        for accuracy in COMPARISON.benchmark.IPM_LINEAR_ACCURACIES:
            if accuracy in ("strict", "blockwise"):
                continue
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", "ipm", "--output-dir", str(self.output), "--ipm-krylov-history-size", "1",
                         "--ipm-linear-accuracy", accuracy]
            with self.subTest(tool="benchmark", accuracy=accuracy), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())
            for side in ("candidate", "reference"):
                prefix = "--" if side == "candidate" else "--reference-"
                other_prefix = "--reference-" if side == "candidate" else "--"
                with self.subTest(side=side, accuracy=accuracy), self.assertRaises(SystemExit) as error, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.run_comparison(default_pdlp=False, extra_args=[
                        "--candidate-backend", "ipm", "--reference-backend", "ipm",
                        prefix + "ipm-krylov-history-size", "1", prefix + "ipm-linear-accuracy", accuracy,
                        other_prefix + "ipm-krylov-history-size", "8", other_prefix + "ipm-linear-accuracy", "strict"])
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.commands)
                self.assertFalse(self.output.exists())

    def test_central_corrections_report_requires_matching_integer(self):
        cases = [(1, "missing"), (1, None), (1, True), (1, 1.0), (1, "1"), (1, 0),
                 (0, False), (0, 0.0), (0, "0"), (0, 1)]
        for index, (requested, reported) in enumerate(cases):
            with self.subTest(requested=requested, reported=reported):
                self.output = self.root / f"bad-corrections-{index}"
                def responder(command, **kwargs):
                    process = self.fake_process(command, **kwargs)
                    value = json.loads(process.stdout)
                    if value["backend"] == "ipm":
                        if reported == "missing":
                            del value["options"]["ipm"]["central_corrections"]
                        else:
                            value["options"]["ipm"]["central_corrections"] = reported
                        process.stdout = json.dumps(value).encode()
                    return process
                status, records, summary = self.run_comparison(responder, default_pdlp=False, repeats=1,
                    extra_args=["--candidate-backend", "ipm", "--ipm-central-corrections", str(requested)])
                self.assertEqual(status, 1)
                self.assertEqual(summary["quality_pass_count"], 2)
                self.assertEqual(summary["variants"][summary["candidate_variant"]]["quality_failure_counts"],
                                 {"ipm_central_corrections_mismatch": 2})
                self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                    if record["requested_backend"] == "ipm"))

    def test_ipm_switch_report_requires_matching_boolean(self):
        responses = [("on", "missing"), ("on", None), ("on", 1), ("on", "on"), ("on", False),
                     ("off", 0), ("off", "off"), ("off", True)]
        cases = [(option, field, requested, reported)
                 for option, field in (("ipm-singleton-elimination", "eliminate_singletons"),
                                       ("ipm-inexact-prefilter", "inexact_prefilter"),
                                       ("ipm-fixed-variable-elimination", "eliminate_fixed_variables"),
                                       ("ipm-singleton-presolve", "presolve_singleton_rows"))
                 for requested, reported in responses]
        for index, (option, field, requested, reported) in enumerate(cases):
            with self.subTest(option=option, requested=requested, reported=reported):
                self.output = self.root / f"bad-boolean-{index}"
                def responder(command, **kwargs):
                    process = self.fake_process(command, **kwargs)
                    value = json.loads(process.stdout)
                    if value["backend"] == "ipm":
                        if reported == "missing":
                            del value["options"]["ipm"][field]
                        else:
                            value["options"]["ipm"][field] = reported
                        process.stdout = json.dumps(value).encode()
                    return process
                status, records, summary = self.run_comparison(responder, repeats=1, default_pdlp=False,
                    extra_args=["--candidate-backend", "ipm", "--ipm-linear-accuracy", "inexact", "--" + option, requested])
                self.assertEqual(status, 1)
                self.assertEqual(summary["variants"][summary["candidate_variant"]]["quality_failure_counts"],
                                 {option.replace("-", "_") + "_mismatch": 2})
                self.assertEqual(summary["quality_pass_count"], 2)
                self.assertTrue(all(record["quality_penalized_wall_seconds"] == 3 for record in records
                                    if record["requested_backend"] == "ipm"))

    def test_benchmark_singleton_and_highs_presolve_flags_are_recorded(self):
        cases = [("ipm", ["--ipm-singleton-elimination", "off"], "ipm_requested_options", "eliminate_singletons", False),
                 ("ipm", ["--ipm-dynamic-regularization", "on"], "ipm_requested_options", "dynamic_regularization", True),
                 ("ipm", ["--ipm-kkt-equilibration", "on"], "ipm_requested_options", "equilibrate_kkt", True),
                 ("ipm", ["--ipm-objective-scaling", "on"], "ipm_requested_options", "objective_scaling", True),
                 ("ipm", ["--ipm-objective-scaling", "off"], "ipm_requested_options", "objective_scaling", False),
                 ("ipm", ["--ipm-method", "mehrotra"], "ipm_requested_options", "method", "mehrotra"),
                 ("ipm", ["--ipm-method", "homogeneous"], "ipm_requested_options", "method", "homogeneous"),
                 ("ipm", ["--ipm-linear-accuracy", "homogeneous_scaled", "--ipm-method", "homogeneous"], "ipm_requested_options", "linear_accuracy", "homogeneous_scaled"),
                 ("ipm", ["--ipm-linear-accuracy", "backward"], "ipm_requested_options", "linear_accuracy", "backward"),
                 ("ipm", ["--ipm-linear-accuracy", "inexact"], "ipm_requested_options", "linear_accuracy", "inexact"),
                 ("ipm", ["--ipm-linear-accuracy", "blockwise"], "ipm_requested_options", "linear_accuracy", "blockwise"),
                 ("ipm", ["--ipm-linear-accuracy", "blockwise_weighted"], "ipm_requested_options", "linear_accuracy", "blockwise_weighted"),
                 ("ipm", ["--ipm-linear-accuracy", "blockwise_forcing"], "ipm_requested_options", "linear_accuracy", "blockwise_forcing"),
                 ("ipm", ["--ipm-linear-accuracy", "blockwise_forcing_weighted"], "ipm_requested_options", "linear_accuracy", "blockwise_forcing_weighted"),
                 ("ipm", ["--ipm-linear-accuracy", "adaptive_regularization"], "ipm_requested_options", "linear_accuracy", "adaptive_regularization"),
                 ("ipm", ["--ipm-linear-accuracy", "primal_dual_regularization"], "ipm_requested_options", "linear_accuracy", "primal_dual_regularization"),
                 ("ipm", ["--ipm-linear-accuracy", "curvature_regularization"], "ipm_requested_options", "linear_accuracy", "curvature_regularization"),
                 ("ipm", ["--ipm-linear-accuracy", "full_newton"], "ipm_requested_options", "linear_accuracy", "full_newton"),
                 ("ipm", ["--ipm-central-corrections", "0"], "ipm_requested_options", "central_corrections", 0),
                 ("ipm", ["--ipm-central-corrections", "1"], "ipm_requested_options", "central_corrections", 1),
                 ("ipm", ["--ipm-direction-retries", "0"], "ipm_requested_options", "direction_retries", 0),
                 ("ipm", ["--ipm-krylov-history-size", "0"], "ipm_requested_options", "krylov_history_size", 0),
                 ("ipm", ["--ipm-krylov-history-size", "8"], "ipm_requested_options", "krylov_history_size", 8),
                 ("ipm", ["--ipm-krylov-history-size", "1", "--ipm-linear-accuracy", "blockwise"], "ipm_requested_options", "krylov_history_size", 1),
                 ("ipm", ["--ipm-krylov-history-size", "0", "--ipm-linear-accuracy", "blockwise_weighted"], "ipm_requested_options", "krylov_history_size", 0),
                 ("ipm", ["--ipm-direction-retries", "8"], "ipm_requested_options", "direction_retries", 8),
                 ("ipm", ["--ipm-direction-retries", "8", "--ipm-linear-accuracy", "blockwise"], "ipm_requested_options", "direction_retries", 8),
                 ("ipm", ["--ipm-direction-retries", "8", "--ipm-linear-accuracy", "blockwise_weighted"], "ipm_requested_options", "direction_retries", 8),
                 ("ipm", ["--ipm-linear-solver", "sparselu"], "ipm_requested_options", "linear_solver", "sparselu"),
                 ("ipm", ["--ipm-linear-solver", "normal", "--ipm-singleton-elimination", "on"], "ipm_requested_options", "linear_solver", "normal"),
                 ("ipm", ["--ipm-linear-solver", "normal_cholmod", "--ipm-singleton-elimination", "on"], "ipm_requested_options", "linear_solver", "normal_cholmod"),
                 ("ipm", ["--ipm-linear-solver", "normal_cholmod_aat", "--ipm-singleton-elimination", "on"], "ipm_requested_options", "linear_solver", "normal_cholmod_aat"),
                 ("ipm", ["--ipm-inexact-prefilter", "on", "--ipm-linear-accuracy", "inexact"], "ipm_requested_options", "inexact_prefilter", True),
                 ("ipm", ["--ipm-inexact-prefilter", "off"], "ipm_requested_options", "inexact_prefilter", False),
                 ("ipm", ["--ipm-fixed-variable-elimination", "on"], "ipm_requested_options", "eliminate_fixed_variables", True),
                 ("ipm", ["--ipm-fixed-variable-elimination", "off"], "ipm_requested_options", "eliminate_fixed_variables", False),
                 ("ipm", ["--ipm-singleton-presolve", "on"], "ipm_requested_options", "presolve_singleton_rows", True),
                 ("ipm", ["--ipm-singleton-presolve", "off"], "ipm_requested_options", "presolve_singleton_rows", False),
                 ("ipm", ["--ipm-presolve-fixed-elimination", "on", "--ipm-singleton-presolve", "on"], "ipm_requested_options", "eliminate_presolved_fixed_variables", True),
                 ("ipm", ["--ipm-presolve-fixed-elimination", "off"], "ipm_requested_options", "eliminate_presolved_fixed_variables", False),
                 ("highs", ["--highs-presolve", "off"], "highs_requested_options", "presolve", "off")]
        for index, (backend, flags, key, field, expected) in enumerate(cases):
            with self.subTest(backend=backend):
                output = self.root / f"benchmark-{index}"
                arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                             "--backend", backend, "--output-dir", str(output), *flags]
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(COMPARISON.benchmark.subprocess, "run", side_effect=self.fake_process), \
                        contextlib.redirect_stdout(io.StringIO()):
                    status = COMPARISON.benchmark.main()
                self.assertEqual(status, 0)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest[key][field], expected)
                record = json.loads((output / "runs.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(record["classification"], "resolved")
                self.assertEqual(record["command"][record["command"].index(flags[0]) + 1], flags[1])

    def test_benchmark_backend_specific_switches_rejected_before_running(self):
        for backend, flags in [("pdlp", ["--ipm-singleton-elimination", "on"]),
                               ("highs", ["--ipm-method", "mehrotra"]),
                               ("pdlp", ["--ipm-method", "homogeneous"]),
                               ("ipm", ["--ipm-method", "unknown"]),
                               ("highs", ["--ipm-singleton-elimination", "off"]),
                               ("pdlp", ["--ipm-dynamic-regularization", "on"]),
                               ("highs", ["--ipm-kkt-equilibration", "on"]),
                               ("highs", ["--ipm-objective-scaling", "on"]),
                               ("pdlp", ["--ipm-objective-scaling", "off"]),
                               ("ipm", ["--ipm-objective-scaling", "invalid"]),
                               ("highs", ["--ipm-linear-accuracy", "backward"]),
                               ("pdlp", ["--ipm-central-corrections", "1"]),
                               ("highs", ["--ipm-central-corrections", "0"]),
                               ("ipm", ["--ipm-central-corrections", "2"]),
                               ("pdlp", ["--ipm-linear-solver", "sparselu"]),
                                ("highs", ["--ipm-linear-solver", "qdldl"]),
                                ("pdlp", ["--ipm-inexact-prefilter", "on"]),
                                ("highs", ["--ipm-inexact-prefilter", "off"]),
                                ("ipm", ["--ipm-inexact-prefilter", "true"]),
                               ("ipm", ["--highs-presolve", "off"]),
                               ("pdlp", ["--highs-presolve", "on"])]:
            arguments = [str(COMPARISON.benchmark.__file__), str(self.models[0]), "--solver", str(self.solver),
                         "--backend", backend, "--output-dir", str(self.output), *flags]
            with self.subTest(backend=backend, flags=flags), mock.patch.object(sys, "argv", arguments), \
                    mock.patch.object(COMPARISON.benchmark.subprocess, "run") as process, \
                    self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                COMPARISON.benchmark.main()
            self.assertEqual(error.exception.code, 2)
            process.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_same_backend_keeps_common_budget_for_existing_ablation(self):
        status, records, summary = self.run_comparison(repeats=1, extra_args=[
            "--reference-pdlp-mode", "adaptive", "--pdlp-mode", "halpern", "--iteration-limit", "12345"])
        self.assertEqual(status, 0)
        self.assertTrue(all(r["report"]["options"]["pdlp"]["max_iterations"] == 12345 for r in records))

    def test_ipm_without_optimality_evidence_or_matching_budget_fails(self):
        calls = 0
        def respond(command, **kwargs):
            nonlocal calls
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            if value["backend"] == "ipm":
                calls += 1
                if calls == 1:
                    value.pop("lp_optimality")
                else:
                    value["options"]["ipm"]["max_iterations"] = 200
                process.stdout = json.dumps(value).encode("utf-8")
            return process
        status, records, summary = self.run_comparison(respond, repeats=1, default_pdlp=False,
                                                      extra_args=["--candidate-backend", "ipm", "--iteration-limit", "123"])
        self.assertEqual(status, 1)
        self.assertEqual(summary["backends"]["ipm"]["quality_failure_counts"],
                         {"invalid_report": 1, "iteration_limit_mismatch": 1})

    def test_incompatible_backend_options_are_rejected_before_running(self):
        for flags in [["--candidate-backend", "ipm", "--pdlp-mode", "adaptive"],
                      ["--ipm-method", "mehrotra"],
                      ["--candidate-backend", "highs", "--ipm-method", "homogeneous"],
                      ["--reference-ipm-method", "homogeneous"],
                      ["--reference-backend", "pdlp", "--reference-ipm-method", "mehrotra"],
                      ["--candidate-backend", "ipm", "--ipm-method", "unknown"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-method", "mehrotra"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--reference-ipm-method", "mehrotra"],
                      ["--candidate-backend", "ipm", "--pdlp-history-size", "0"],
                      ["--reference-backend", "ipm", "--reference-pdlp-mode", "halpern"],
                      ["--reference-iteration-limit", "100"],
                      ["--candidate-backend", "highs", "--iteration-limit", "100"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm"],
                      ["--ipm-singleton-elimination", "on"],
                      ["--candidate-backend", "highs", "--ipm-singleton-elimination", "off"],
                      ["--reference-ipm-singleton-elimination", "on"],
                      ["--reference-backend", "pdlp", "--reference-ipm-singleton-elimination", "off"],
                      ["--candidate-backend", "ipm", "--ipm-singleton-elimination", "true"],
                      ["--ipm-dynamic-regularization", "on"],
                      ["--reference-ipm-dynamic-regularization", "on"],
                      ["--ipm-kkt-equilibration", "on"],
                      ["--reference-ipm-kkt-equilibration", "off"],
                      ["--ipm-objective-scaling", "off"],
                      ["--reference-ipm-objective-scaling", "on"],
                      ["--candidate-backend", "ipm", "--ipm-objective-scaling", "invalid"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-objective-scaling", "on"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--reference-ipm-objective-scaling", "on"],
                      ["--ipm-linear-accuracy", "backward"],
                      ["--reference-ipm-linear-accuracy", "strict"],
                      ["--candidate-backend", "ipm", "--ipm-linear-accuracy", "loose"],
                      ["--ipm-central-corrections", "1"],
                      ["--reference-ipm-central-corrections", "0"],
                      ["--candidate-backend", "ipm", "--ipm-central-corrections", "2"],
                      ["--reference-backend", "ipm", "--reference-ipm-central-corrections", "-1"],
                      ["--candidate-backend", "ipm", "--ipm-central-corrections", "true"],
                      ["--ipm-linear-solver", "sparselu"],
                      ["--reference-ipm-linear-solver", "qdldl"],
                       ["--candidate-backend", "ipm", "--ipm-linear-solver", "LU"],
                       ["--ipm-inexact-prefilter", "on"],
                       ["--reference-ipm-inexact-prefilter", "off"],
                       ["--candidate-backend", "ipm", "--ipm-inexact-prefilter", "true"],
                       ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-inexact-prefilter", "off"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-linear-solver", "qdldl"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-central-corrections", "0"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-singleton-elimination", "off"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-dynamic-regularization", "off"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-kkt-equilibration", "off"],
                      ["--candidate-backend", "ipm", "--reference-backend", "ipm", "--ipm-linear-accuracy", "strict"],
                      ["--candidate-backend", "highs", "--highs-presolve", "choose"]]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
                self.run_comparison(extra_args=flags, default_pdlp=False)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.commands)
            self.assertFalse(self.output.exists())

    def test_existing_output_and_duplicate_list_are_rejected(self):
        self.run_comparison(repeats=1)
        original = (self.output / "runs.jsonl").read_bytes()
        with self.assertRaises(SystemExit) as error, contextlib.redirect_stderr(io.StringIO()):
            self.run_comparison(repeats=1)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((self.output / "runs.jsonl").read_bytes(), original)
        self.model_list.write_text(self.models[0].name + "\n" + self.models[0].name + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            COMPARISON.load_models(self.model_list)

    def test_speedups_use_medians_only_after_all_repeats_pass(self):
        models, _ = COMPARISON.load_models(self.model_list)
        records = []
        for model in models:
            for backend, times in [("highs", [2.0, 100.0, 3.0]), ("pdlp", [1.0, 2.0, 1.5])]:
                for value in times:
                    records.append({"model_index": model["model_index"], "requested_backend": backend,
                                    "quality_pass": True, "quality_failure": None, "status": "optimal",
                                    "elapsed_wall_seconds": value, "quality_penalized_wall_seconds": min(value, 20),
                                    "quality_penalized_numerical_seconds": min(value, 10),
                                    "report": {"timings": {"solve_seconds": value, "verification_seconds": 0.0}}})
        records[-1]["quality_pass"] = False
        records[-1]["quality_failure"] = "wall_timeout"
        summary = COMPARISON.summarize(records, models, 3, 20, 10)
        self.assertEqual(summary["models"][0]["wall_speedup_highs_over_pdlp"], 2.0)
        self.assertEqual(summary["models"][0]["wall_winner"], "pdlp")
        self.assertIsNone(summary["models"][1]["wall_speedup_highs_over_pdlp"])
        self.assertEqual(summary["wall_winner_counts"], {"pdlp": 1})

    def test_repeated_verification_cost_is_included_in_secondary_time(self):
        def respond(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            value["timings"]["solve_seconds"] = 1.0 if value["backend"] == "highs" else 0.1
            value["timings"]["verification_seconds"] = 0.1 if value["backend"] == "highs" else 1.9
            value["timings"]["total_seconds"] = value["timings"]["solve_seconds"] + value["timings"]["verification_seconds"] + value["timings"]["read_seconds"]
            process.stdout = json.dumps(value).encode("utf-8")
            return process

        status, records, summary = self.run_comparison(respond, repeats=1)
        self.assertEqual(status, 0)
        self.assertEqual(records[0]["numerical_seconds"], 1.1)
        self.assertEqual(records[1]["numerical_seconds"], 2.0)
        self.assertEqual(summary["models"][0]["numerical_speedup_highs_over_pdlp"], 0.55)
        self.assertEqual(summary["models"][0]["median_numerical_seconds"], {"highs": 1.1, "pdlp": 2.0})

    def test_driver_json_validation_is_outside_wall_measurement(self):
        current_time = [10.0]
        options = COMPARISON.argparse.Namespace(solver=self.solver, backend="highs", time_limit=2,
                                                wall_timeout=3, threads=1, mip_gap=1e-4,
                                                iteration_limit=None, scaling_passes=None, pdlp_mode=None)
        original_parser = COMPARISON.benchmark.strict_json

        def process(command, **kwargs):
            current_time[0] = 11.0
            return self.fake_process(command, **kwargs)

        def slow_parser(value):
            current_time[0] = 20.0
            return original_parser(value)

        with mock.patch.object(COMPARISON.benchmark.subprocess, "run", side_effect=process), \
                mock.patch.object(COMPARISON.benchmark.time, "perf_counter", side_effect=lambda: current_time[0]), \
                mock.patch.object(COMPARISON.benchmark, "strict_json", side_effect=slow_parser):
            record = COMPARISON.benchmark.run_one(options, self.models[0], 0, 0)
        self.assertEqual(record["classification"], "resolved")
        self.assertEqual(record["elapsed_wall_seconds"], 1.0)

    def test_wall_budget_overrun_is_not_counted_as_success(self):
        current_time = [10.0]

        def process(command, **kwargs):
            current_time[0] += 4.0
            return self.fake_process(command, **kwargs)

        with mock.patch.object(COMPARISON.benchmark.time, "perf_counter", side_effect=lambda: current_time[0]):
            status, records, summary = self.run_comparison(process, repeats=1)
        self.assertEqual(status, 1)
        self.assertTrue(all(record["quality_failure"] == "wall_budget_exceeded" for record in records))
        self.assertEqual(summary["quality_pass_count"], 0)
        self.assertEqual(summary["backends"]["highs"]["wall_budget_exceeded_count"], 2)
        self.assertTrue(all(model["wall_speedup_highs_over_pdlp"] is None for model in summary["models"]))

    def test_numerical_budget_including_verification_is_strict(self):
        def respond(command, **kwargs):
            process = self.fake_process(command, **kwargs)
            value = json.loads(process.stdout)
            value["timings"].update(solve_seconds=1.9, verification_seconds=0.2, total_seconds=2.11)
            process.stdout = json.dumps(value).encode("utf-8")
            return process
        status, records, summary = self.run_comparison(respond, repeats=1)
        self.assertEqual(status, 1)
        self.assertEqual(summary["quality_pass_count"], 0)
        self.assertTrue(all(record["quality_failure"] == "numerical_budget_exceeded" for record in records))
        self.assertTrue(all(record["numerical_seconds"] == 2.1 for record in records))
        self.assertTrue(all(record["quality_penalized_numerical_seconds"] == 2 for record in records))
        self.assertTrue(all(model["median_wall_seconds"] is None for model in summary["models"]))


if __name__ == "__main__":
    unittest.main()
