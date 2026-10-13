#!/usr/bin/env python3
"""检查命令行协议、结果方向与基准失败留存。"""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SOLVER = Path(sys.argv[1]).resolve()
BENCHMARK = Path(sys.argv[2]).resolve()
sys.argv = [sys.argv[0]]


def reject_constant(value):
    raise ValueError("非法 JSON 常量：" + value)


def parse_json(value):
    return json.loads(value, parse_constant=reject_constant)


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="solver cli test ")
        cls.root = Path(cls.temporary.name)
        cls.minimum = cls.root / "minimum with spaces.lp"
        cls.minimum.write_text("Minimize\n obj: x + y\nSubject To\n r1: 2 x + y >= 4\n r2: x + 2 y >= 4\nBounds\n 0 <= x <= 10\n 0 <= y <= 10\nEnd\n", encoding="utf-8")
        cls.maximum = cls.root / "maximum.lp"
        cls.maximum.write_text("Maximize\n obj: x + 2 y\nSubject To\n r1: x + y <= 4\nBounds\n 0 <= x <= 2\n 0 <= y <= 3\nEnd\n", encoding="utf-8")
        cls.maximum_offset = cls.root / "maximum with offset.lp"
        cls.maximum_offset.write_text("Maximize\n obj: x + 2 y + 5\nSubject To\n r1: x + y <= 4\nBounds\n 0 <= x <= 2\n 0 <= y <= 3\nEnd\n", encoding="utf-8")
        cls.integer = cls.root / "integer.lp"
        cls.integer.write_text("Maximize\n obj: 3 x + 2 y\nSubject To\n r1: x + y <= 2.5\nBounds\n 0 <= x <= 2\n 0 <= y <= 2\nGenerals\n x y\nEnd\n", encoding="utf-8")
        cls.infeasible = cls.root / "infeasible.lp"
        cls.infeasible.write_text("Minimize\n obj: x\nSubject To\n r1: x >= 2\n r2: x <= 1\nEnd\n", encoding="utf-8")
        cls.unbounded = cls.root / "unbounded.lp"
        cls.unbounded.write_text("Minimize\n obj: - x\nSubject To\n r1: y = 0\nBounds\n x >= 0\n y free\nEnd\n", encoding="utf-8")
        specification = importlib.util.spec_from_file_location("solver_benchmark", BENCHMARK)
        cls.benchmark = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.benchmark)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def cli(self, *args):
        process = subprocess.run([str(SOLVER), *map(str, args)], capture_output=True,
                                 timeout=30, check=False)
        process.stderr = process.stderr.decode("utf-8", errors="replace")
        try:
            process.stdout = process.stdout.decode("utf-8", errors="strict")
            report = parse_json(process.stdout)
        except (ValueError, TypeError) as error:
            self.fail(f"CLI 未输出合法 JSON：{error}\n{process.stdout}\n{process.stderr}")
        return process, report

    def bench(self, *args):
        process = subprocess.run([sys.executable, str(BENCHMARK), *map(str, args)],
                                 capture_output=True, timeout=60, check=False)
        process.stdout = process.stdout.decode("utf-8", errors="strict")
        process.stderr = process.stderr.decode("utf-8", errors="replace")
        return process

    def test_help_and_version_are_json(self):
        process, version = self.cli("--version")
        self.assertEqual(process.returncode, 0)
        self.assertEqual(version["backend"], "highs")
        self.assertTrue(version["backend_version"])
        self.assertEqual(version["supported_backends"], ["highs", "pdlp", "ipm"])
        process, help_report = self.cli("--help")
        self.assertEqual(process.returncode, 0)
        self.assertIn("usage", help_report)
        self.assertEqual(help_report["defaults"]["ipm"]["max_iterations"], 200)
        self.assertEqual(help_report["defaults"]["ipm"]["scaling_passes"], 5)
        self.assertIs(help_report["defaults"]["ipm"]["presolve_singleton_rows"], False)
        self.assertIs(help_report["defaults"]["ipm"]["eliminate_presolved_fixed_variables"], False)
        self.assertIs(type(help_report["defaults"]["ipm"]["krylov_history_size"]), int)
        self.assertEqual(help_report["defaults"]["ipm"]["krylov_history_size"], 0)

    def test_minimum_and_path_with_spaces(self):
        process, report = self.cli("solve", self.minimum, "--threads", 1, "--seed", 42)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["status"], "optimal")
        self.assertEqual(report["model"]["sense"], "minimize")
        self.assertAlmostEqual(report["objective"], 8 / 3)
        self.assertTrue(report["verification"]["feasible"])
        self.assertTrue(report["verification"]["objective_consistent"])
        self.assertEqual(report["options"]["seed"], 42)
        self.assertEqual(report["options"]["threads"], 1)
        self.assertEqual(Path(report["input"]), self.minimum)
        self.assertEqual(len(report["solution"]), 2)

    def test_highs_fixed_lp_algorithm_and_crossover(self):
        for algorithm, crossover in [("choose", "on"), ("simplex", "on"), ("ipm", "off")]:
            with self.subTest(algorithm=algorithm, crossover=crossover):
                presolve = "off" if algorithm == "ipm" and crossover == "off" else "choose"
                process, result = self.cli("solve", self.maximum_offset, "--highs-solver", algorithm,
                                           "--highs-crossover", crossover, "--highs-presolve", presolve,
                                           "--backend", "highs")
                self.assertEqual(process.returncode, 0, result)
                self.assertEqual(result["options"]["highs"], {"solver": algorithm, "run_crossover": crossover, "presolve": presolve})
                self.assertTrue(result["lp_optimality"]["numerically_optimal"])
                self.assertAlmostEqual(result["objective"], 12, places=5)

    def test_highs_specific_options_cannot_be_silently_ignored(self):
        for arguments in [("--backend", "ipm", "--highs-solver", "choose"),
                          ("--highs-crossover", "on", "--backend", "pdlp"),
                          ("--highs-solver", "unknown"), ("--highs-crossover", "unknown")]:
            with self.subTest(arguments=arguments):
                process, result = self.cli("solve", self.minimum, *arguments)
                self.assertEqual(process.returncode, 2, result)
        process, result = self.cli("solve", self.integer, "--highs-solver", "ipm")
        self.assertEqual(process.returncode, 2, result)

    def test_ipm_singleton_elimination_is_explicit_and_reported(self):
        for value in ["off", "on"]:
            process, result = self.cli("solve", self.minimum, "--ipm-singleton-elimination", value, "--backend", "ipm")
            self.assertEqual(process.returncode, 0, result)
            self.assertIs(result["options"]["ipm"]["eliminate_singletons"], value == "on")
            self.assertIsInstance(result["ipm_kkt_dimension"], int)
            self.assertIsInstance(result["ipm_factor_nonzeros"], int)
            self.assertGreater(result["ipm_factor_nonzeros"], 0)
            self.assertEqual(result["ipm_eliminated_singletons"] > 0, value == "on")
        for backend in ["highs", "pdlp"]:
            process, result = self.cli("solve", self.minimum, "--backend", backend, "--ipm-singleton-elimination", "off")
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_dynamic_regularization_is_explicit_and_reported(self):
        for value in ["off", "on"]:
            process, result = self.cli("solve", self.minimum, "--ipm-dynamic-regularization", value,
                                       "--ipm-singleton-elimination", "on", "--backend", "ipm")
            self.assertEqual(process.returncode, 0, result)
            self.assertIs(result["options"]["ipm"]["dynamic_regularization"], value == "on")
            self.assertIsInstance(result["ipm_regularized_pivots"], int)
            self.assertGreaterEqual(result["ipm_regularized_pivots"], 0)
            if value == "off":
                self.assertEqual(result["ipm_regularized_pivots"], 0)
        for backend in ["highs", "pdlp"]:
            process, result = self.cli("solve", self.minimum, "--backend", backend,
                                       "--ipm-dynamic-regularization", "off")
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_kkt_equilibration_is_explicit_and_reported(self):
        for value in ["off", "on"]:
            process, result = self.cli("solve", self.minimum, "--ipm-kkt-equilibration", value,
                                       "--ipm-singleton-elimination", "on", "--backend", "ipm")
            self.assertEqual(process.returncode, 0, result)
            self.assertIs(result["options"]["ipm"]["equilibrate_kkt"], value == "on")
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
        for arguments in [("--backend", "highs", "--ipm-kkt-equilibration", "off"),
                          ("--ipm-kkt-equilibration", "off", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-kkt-equilibration", "invalid")]:
            process, result = self.cli("solve", self.minimum, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_linear_accuracy_keeps_original_optimality_checks(self):
        for value in ["strict", "backward", "inexact", "blockwise", "blockwise_weighted", "blockwise_forcing", "blockwise_forcing_weighted", "adaptive_regularization", "primal_dual_regularization", "curvature_regularization"]:
            process, result = self.cli("solve", self.minimum, "--ipm-linear-accuracy", value,
                                       "--ipm-kkt-equilibration", "on", "--backend", "ipm")
            self.assertEqual(process.returncode, 0, result)
            self.assertEqual(result["options"]["ipm"]["linear_accuracy"], value)
            self.assertEqual(result["options"]["ipm"]["direction_retries"], 0)
            self.assertEqual(result["options"]["ipm"]["krylov_history_size"], 0)
            for field in ("ipm_krylov_recycle_projections", "ipm_krylov_recycle_acceptances",
                          "ipm_krylov_recycle_vectors", "ipm_krylov_recycle_peak_dimension"):
                self.assertIs(type(result[field]), int)
                self.assertEqual(result[field], 0)
            self.assertEqual(result["ipm_direction_retries"], 0)
            self.assertEqual(result["ipm_direction_retry_successes"], 0)
            self.assertGreaterEqual(result["ipm_unshifted_factorizations"], 0)
            self.assertLessEqual(result["ipm_unshifted_factorizations"], result["ipm_factorizations"])
            self.assertGreater(result["ipm_primal_regularization"], 0)
            ratio = result["ipm_dual_regularization"] / result["ipm_primal_regularization"]
            self.assertEqual(ratio, 8192 if value == "primal_dual_regularization" else 1)
            self.assertIsInstance(result["ipm_regularization_searches"], int)
            self.assertGreaterEqual(result["ipm_regularization_reductions"], 0)
            self.assertLessEqual(result["ipm_regularization_reductions"], result["ipm_regularization_searches"])
            self.assertLessEqual(result["ipm_regularization_searches"], result["ipm_iterations"] + 1)
            if value != "adaptive_regularization":
                self.assertEqual(result["ipm_regularization_searches"], 0)
            self.assertIsInstance(result["ipm_weighted_krylov_steps"], int)
            self.assertGreaterEqual(result["ipm_weighted_krylov_steps"], 0)
            self.assertLessEqual(result["ipm_weighted_krylov_steps"], result["ipm_refinement_steps"])
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
            self.assertIsInstance(result["ipm_backward_error_acceptances"], int)
            self.assertGreaterEqual(result["ipm_backward_error_acceptances"], 0)
            if value != "backward":
                self.assertEqual(result["ipm_backward_error_acceptances"], 0)
            self.assertIsInstance(result["ipm_inexact_acceptances"], int)
            self.assertGreaterEqual(result["ipm_inexact_acceptances"], 0)
            if value != "inexact":
                self.assertEqual(result["ipm_inexact_acceptances"], 0)
        for arguments in [("--backend", "highs", "--ipm-linear-accuracy", "strict"),
                          ("--ipm-linear-accuracy", "backward", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-linear-accuracy", "invalid"),
                          ("--backend", "ipm", "--ipm-linear-accuracy", "inexact",
                           "--ipm-central-corrections", "1")]:
            process, result = self.cli("solve", self.minimum, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_full_newton_preserves_original_checks_and_work_counts(self):
        fields = ("ipm_full_newton_preconditioner_solves", "ipm_full_newton_products")
        for linear in ("qdldl", "normal_augmented"):
            for budget in (0, 1, 8):
                with self.subTest(linear=linear, retries=budget):
                    process, result = self.cli("solve", self.minimum, "--backend", "ipm",
                                               "--ipm-linear-accuracy", "full_newton", "--ipm-linear-solver", linear,
                                               "--ipm-direction-retries", budget, "--ipm-krylov-history-size", "0",
                                               "--ipm-central-corrections", "0", "--ipm-kkt-equilibration", "on")
                    self.assertEqual(process.returncode, 0, result)
                    self.assertEqual(result["options"]["ipm"]["linear_accuracy"], "full_newton")
                    self.assertEqual(result["options"]["ipm"]["direction_retries"], budget)
                    self.assertAlmostEqual(result["objective"], 8 / 3, places=5)
                    self.assertTrue(result["verification"]["feasible"])
                    self.assertTrue(result["lp_optimality"]["numerically_optimal"])
                    for field in fields:
                        self.assertIs(type(result[field]), int)
                        self.assertGreater(result[field], 0)
        for backend in ("ipm", "highs", "pdlp"):
            process, result = self.cli("solve", self.minimum, "--backend", backend)
            self.assertEqual(process.returncode, 0, result)
            for field in fields:
                if backend == "ipm":
                    self.assertIs(type(result[field]), int)
                    self.assertEqual(result[field], 0)
                else:
                    self.assertIsNone(result[field])

    def test_ipm_full_newton_rejects_unsupported_combinations(self):
        cases = [("--backend", backend) for backend in ("highs", "pdlp")]
        cases += [("--backend", "ipm", flag, "1") for flag in
                  ("--ipm-central-corrections", "--ipm-krylov-history-size")]
        cases += [("--backend", "ipm", "--ipm-inexact-prefilter", "on")]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process, result = self.cli("solve", self.minimum, "--ipm-linear-accuracy", "full_newton", *arguments)
                self.assertEqual(process.returncode, 2, result)
                self.assertEqual(result["status"], "error")

    def test_ipm_inexact_prefilter_is_explicit_and_restricted(self):
        for value in ["on", "off"]:
            process, result = self.cli("solve", self.minimum, "--backend", "ipm",
                                       "--ipm-linear-accuracy", "inexact",
                                       "--ipm-inexact-prefilter", value)
            self.assertEqual(process.returncode, 0, result)
            self.assertIs(result["options"]["ipm"]["inexact_prefilter"], value == "on")
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
            self.assertIsInstance(result["ipm_inexact_prefilter_rejections"], int)
            self.assertGreaterEqual(result["ipm_inexact_prefilter_rejections"], 0)
            if value == "off":
                self.assertEqual(result["ipm_inexact_prefilter_rejections"], 0)
        for arguments in [("--backend", "highs", "--ipm-inexact-prefilter", "off"),
                          ("--ipm-inexact-prefilter", "off", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-inexact-prefilter", "invalid"),
                          ("--backend", "ipm", "--ipm-inexact-prefilter", "on")]:
            process, result = self.cli("solve", self.minimum, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_central_corrections_have_explicit_budget(self):
        for value in ["0", "1"]:
            process, result = self.cli("solve", self.minimum, "--ipm-central-corrections", value,
                                       "--backend", "ipm")
            self.assertEqual(process.returncode, 0, result)
            self.assertEqual(result["options"]["ipm"]["central_corrections"], int(value))
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
            self.assertIsInstance(result["ipm_central_correction_attempts"], int)
            self.assertIsInstance(result["ipm_central_correction_acceptances"], int)
            self.assertGreaterEqual(result["ipm_central_correction_attempts"],
                                    result["ipm_central_correction_acceptances"])
            if value == "0":
                self.assertEqual(result["ipm_central_correction_attempts"], 0)
        for arguments in [("--backend", "highs", "--ipm-central-corrections", "0"),
                          ("--ipm-central-corrections", "1", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-central-corrections", "2"),
                          ("--backend", "ipm", "--ipm-central-corrections", "-1")]:
            process, result = self.cli("solve", self.minimum, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_direction_retries_have_explicit_budget_and_counts(self):
        for accuracy in ("strict", "primal_dual_regularization", "curvature_regularization", "blockwise", "blockwise_weighted"):
            for budget, corrections in ((0, 0), (1, 0), (8, 1)):
                with self.subTest(accuracy=accuracy, budget=budget, corrections=corrections):
                    process, result = self.cli("solve", self.minimum, "--ipm-direction-retries", budget,
                                               "--backend", "ipm", "--ipm-linear-accuracy", accuracy,
                                               "--ipm-central-corrections", corrections)
                    self.assertEqual(process.returncode, 0, result)
                    self.assertEqual(result["status"], "optimal")
                    self.assertIs(type(result["options"]["ipm"]["direction_retries"]), int)
                    self.assertEqual(result["options"]["ipm"]["direction_retries"], budget)
                    self.assertEqual(result["options"]["ipm"]["central_corrections"], corrections)
                    self.assertAlmostEqual(result["objective"], 8 / 3, places=5)
                    self.assertTrue(result["verification"]["feasible"])
                    self.assertTrue(result["lp_optimality"]["numerically_optimal"])
                    for field in ("ipm_direction_retries", "ipm_direction_retry_successes"):
                        self.assertIs(type(result[field]), int)
                        self.assertGreaterEqual(result[field], 0)
                        if budget == 0:
                            self.assertEqual(result[field], 0)
                    self.assertLessEqual(result["ipm_direction_retry_successes"], result["ipm_direction_retries"])

    def test_ipm_direction_retries_reject_invalid_values_and_modes(self):
        cases = [("--backend", "ipm", "--ipm-direction-retries", value) for value in ("-1", "9", "1.5")]
        cases += [("--ipm-direction-retries", "0", "--backend", backend) for backend in ("highs", "pdlp")]
        cases += [("--backend", "ipm", "--ipm-direction-retries", "1", "--ipm-linear-accuracy", accuracy)
                  for accuracy in ("backward", "inexact", "blockwise_forcing", "blockwise_forcing_weighted",
                                   "adaptive_regularization")]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process, result = self.cli("solve", self.minimum, *arguments)
                self.assertEqual(process.returncode, 2, result)
                self.assertEqual(result["status"], "error")

    def test_ipm_krylov_history_budget_and_counters(self):
        fields = ("ipm_krylov_recycle_projections", "ipm_krylov_recycle_acceptances",
                  "ipm_krylov_recycle_vectors", "ipm_krylov_recycle_peak_dimension")
        for accuracy in ("strict", "blockwise"):
            for budget, corrections, retries in ((0, 0, 0), (1, 0, 0), (8, 1, 8)):
                with self.subTest(accuracy=accuracy, budget=budget):
                    process, result = self.cli("solve", self.minimum, "--backend", "ipm",
                                               "--ipm-linear-accuracy", accuracy, "--ipm-krylov-history-size", budget,
                                               "--ipm-central-corrections", corrections, "--ipm-direction-retries", retries)
                    self.assertEqual(process.returncode, 0, result)
                    self.assertEqual(result["status"], "optimal")
                    self.assertIs(type(result["options"]["ipm"]["krylov_history_size"]), int)
                    self.assertEqual(result["options"]["ipm"]["krylov_history_size"], budget)
                    self.assertAlmostEqual(result["objective"], 8 / 3, places=5)
                    self.assertTrue(result["verification"]["feasible"])
                    self.assertTrue(result["lp_optimality"]["numerically_optimal"])
                    for field in fields:
                        self.assertIs(type(result[field]), int)
                        self.assertGreaterEqual(result[field], 0)
                        if budget == 0:
                            self.assertEqual(result[field], 0)
                    self.assertLessEqual(result["ipm_krylov_recycle_peak_dimension"], budget)
                    self.assertLessEqual(result["ipm_krylov_recycle_acceptances"], result["ipm_krylov_recycle_projections"])
        for backend in ("highs", "pdlp"):
            process, result = self.cli("solve", self.minimum, "--backend", backend)
            self.assertEqual(process.returncode, 0, result)
            for field in fields:
                self.assertIsNone(result[field])

    def test_ipm_krylov_history_rejects_invalid_values_and_positive_modes(self):
        cases = [("--backend", "ipm", "--ipm-krylov-history-size", value) for value in ("-1", "9", "1.5", "true")]
        cases += [("--backend", backend, "--ipm-krylov-history-size", value)
                  for backend in ("highs", "pdlp") for value in ("0", "1")]
        unsupported = ("backward", "inexact", "blockwise_weighted", "blockwise_forcing", "blockwise_forcing_weighted",
                       "adaptive_regularization", "primal_dual_regularization", "curvature_regularization")
        cases += [("--backend", "ipm", "--ipm-krylov-history-size", "1", "--ipm-linear-accuracy", accuracy)
                  for accuracy in unsupported]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process, result = self.cli("solve", self.minimum, *arguments)
                self.assertEqual(process.returncode, 2, result)
                self.assertEqual(result["status"], "error")
        for accuracy in unsupported:
            with self.subTest(zero_budget_accuracy=accuracy):
                process, result = self.cli("solve", self.minimum, "--backend", "ipm",
                                           "--ipm-krylov-history-size", "0", "--ipm-linear-accuracy", accuracy)
                self.assertEqual(process.returncode, 0, result)
                self.assertEqual(result["options"]["ipm"]["krylov_history_size"], 0)
                self.assertTrue(result["lp_optimality"]["numerically_optimal"])

    def test_ipm_fixed_variable_elimination_restores_original_solution(self):
        fixed = self.root / "fixed variables.lp"
        fixed.write_text("Minimize\n obj: 4 x + y - 2 z + 11\nSubject To\n r1: 2 x + y - z >= -3\n r2: x + z = 1\nBounds\n x = -2\n 0 <= y <= 10\n z = 3\nEnd\n", encoding="utf-8")
        for value in ["off", "on"]:
            process, result = self.cli("solve", fixed, "--backend", "ipm", "--ipm-fixed-variable-elimination", value)
            self.assertEqual(process.returncode, 0, result)
            self.assertIs(result["options"]["ipm"]["eliminate_fixed_variables"], value == "on")
            self.assertEqual(result["ipm_eliminated_fixed_variables"], 2 if value == "on" else 0)
            self.assertEqual(result["model"]["columns"], 3)
            self.assertAlmostEqual(result["objective"], 1, places=5)
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
        for arguments in [("--backend", "highs", "--ipm-fixed-variable-elimination", "off"),
                          ("--ipm-fixed-variable-elimination", "on", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-fixed-variable-elimination", "invalid")]:
            process, result = self.cli("solve", fixed, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_ipm_singleton_presolve_merges_rows_and_preserves_original_report(self):
        cases = [("single", " r1: -2 x <= -6\n", 1, 3),
                 ("two", " r1: 2 x >= 4\n r2: -3 x >= -18\n", 2, 2)]
        for name, rows, row_count, objective in cases:
            model = self.root / ("singleton presolve " + name + ".lp")
            model.write_text("Minimize\n obj: x\nSubject To\n" + rows + "Bounds\n 0 <= x <= 10\nEnd\n", encoding="utf-8")
            for value in (None, "off", "on"):
                with self.subTest(model=name, value=value):
                    flags = [] if value is None else ["--ipm-singleton-presolve", value]
                    process, result = self.cli("solve", model, "--backend", "ipm", *flags)
                    self.assertEqual(process.returncode, 0, result)
                    self.assertIs(result["options"]["ipm"]["presolve_singleton_rows"], value == "on")
                    self.assertIs(type(result["ipm_presolved_singleton_rows"]), int)
                    self.assertEqual(result["ipm_presolved_singleton_rows"], row_count if value == "on" else 0)
                    self.assertEqual(result["model"]["rows"], row_count)
                    self.assertEqual(len(result["row_dual"]), row_count)
                    self.assertAlmostEqual(result["objective"], objective, places=5)
                    self.assertTrue(result["verification"]["feasible"])
                    self.assertTrue(result["lp_optimality"]["numerically_optimal"])
        for backend in ("highs", "pdlp"):
            process, result = self.cli("solve", self.minimum, "--backend", backend)
            self.assertEqual(process.returncode, 0, result)
            self.assertIsNone(result["ipm_presolved_singleton_rows"])
            self.assertIsNone(result["ipm_presolved_fixed_columns"])
            for value in ("on", "off"):
                process, result = self.cli("solve", self.minimum, "--ipm-singleton-presolve", value, "--backend", backend)
                self.assertEqual(process.returncode, 2, result)
        process, result = self.cli("solve", self.minimum, "--backend", "ipm", "--ipm-singleton-presolve", "true")
        self.assertEqual(process.returncode, 2, result)

    def test_ipm_presolve_fixed_elimination_restores_original_solution(self):
        cases = [("new", "Minimize\n obj: 3 x + 1\nSubject To\n r1: 2 x >= 4\n r2: -x >= -2\nBounds\n 0 <= x <= 10\nEnd\n", "off", 0, 2, [2], 7),
                 ("combined", "Minimize\n obj: 2 x + y + 5\nSubject To\n r1: 3 y >= 6\n r2: -y >= -2\n r3: x + y >= 3\nBounds\n x = 1\n 0 <= y <= 10\nEnd\n", "on", 1, 3, [1, 2], 9)]
        for name, text, old_fixed, old_count, rows, solution, objective in cases:
            model = self.root / ("presolved fixed " + name + ".lp")
            model.write_text(text, encoding="utf-8")
            for value in (None, "off", "on"):
                with self.subTest(model=name, value=value):
                    flags = [] if value is None else ["--ipm-presolve-fixed-elimination", value]
                    process, result = self.cli("solve", model, "--backend", "ipm", "--ipm-singleton-presolve", "on",
                                               "--ipm-fixed-variable-elimination", old_fixed, *flags)
                    self.assertEqual(process.returncode, 0, result)
                    self.assertIs(result["options"]["ipm"]["eliminate_presolved_fixed_variables"], value == "on")
                    self.assertIs(type(result["ipm_presolved_fixed_columns"]), int)
                    self.assertEqual(result["ipm_presolved_fixed_columns"], 1 if value == "on" else 0)
                    self.assertEqual(result["ipm_eliminated_fixed_variables"], old_count + (1 if value == "on" else 0))
                    self.assertLessEqual(result["ipm_presolved_fixed_columns"], result["ipm_eliminated_fixed_variables"])
                    self.assertEqual(result["model"]["columns"], len(solution))
                    self.assertEqual(result["model"]["rows"], rows)
                    self.assertEqual(len(result["row_dual"]), rows)
                    self.assertEqual(len(result["solution"]), len(solution))
                    for actual, expected in zip(result["solution"], solution):
                        self.assertAlmostEqual(actual, expected, places=5)
                    self.assertAlmostEqual(result["objective"], objective, places=5)
                    self.assertTrue(result["verification"]["feasible"])
                    self.assertTrue(result["lp_optimality"]["numerically_optimal"])

    def test_ipm_presolve_fixed_elimination_requires_matching_backend_and_presolve(self):
        cases = [("--backend", backend, "--ipm-presolve-fixed-elimination", value)
                 for backend in ("highs", "pdlp") for value in ("on", "off")]
        cases += [("--backend", "ipm", "--ipm-presolve-fixed-elimination", "on"),
                  ("--backend", "ipm", "--ipm-singleton-presolve", "off", "--ipm-presolve-fixed-elimination", "on"),
                  ("--backend", "ipm", "--ipm-singleton-presolve", "on", "--ipm-presolve-fixed-elimination", "true")]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process, result = self.cli("solve", self.minimum, *arguments)
                self.assertEqual(process.returncode, 2, result)
        process, result = self.cli("solve", self.minimum, "--backend", "ipm", "--ipm-presolve-fixed-elimination", "off")
        self.assertEqual(process.returncode, 0, result)
        self.assertIs(result["options"]["ipm"]["presolve_singleton_rows"], False)
        self.assertIs(result["options"]["ipm"]["eliminate_presolved_fixed_variables"], False)
        self.assertEqual(result["ipm_presolved_fixed_columns"], 0)

    def test_ipm_curvature_attempt_is_reported_with_singletons(self):
        process, result = self.cli("solve", self.minimum, "--backend", "ipm",
                                   "--ipm-singleton-elimination", "on",
                                   "--ipm-linear-accuracy", "curvature_regularization")
        self.assertEqual(process.returncode, 0, result)
        self.assertGreater(result["ipm_unshifted_factorizations"], 0)
        self.assertLessEqual(result["ipm_unshifted_factorizations"], result["ipm_factorizations"])
        self.assertEqual(result["ipm_primal_regularization"], 0)
        self.assertGreater(result["ipm_dual_regularization"], 0)
        self.assertTrue(result["lp_optimality"]["numerically_optimal"])

    def test_ipm_linear_solver_selection_is_explicit(self):
        for value in ["qdldl", "sparselu", "normal", "qdldl_extended", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined"]:
            process, result = self.cli("solve", self.minimum, "--ipm-linear-solver", value,
                                       "--backend", "ipm", "--ipm-singleton-elimination", "on")
            if value in ("qdldl_extended", "normal_extended", "normal_refined") and process.returncode == 2:
                self.assertIn("本平台 long double 未提供比 double 更高的有效精度", result["message"])
                continue
            self.assertEqual(process.returncode, 0, result)
            self.assertEqual(result["options"]["ipm"]["linear_solver"], value)
            self.assertTrue(result["lp_optimality"]["numerically_optimal"])
            self.assertGreater(result["ipm_factor_nonzeros"], 0)
            self.assertEqual(result["ipm_factor_backend"], "normal" if value == "normal_augmented" else value)
            self.assertEqual(result["ipm_structural_fallbacks"], 0)
            for field in ["ipm_schur_residual_products", "ipm_schur_refinement_steps", "ipm_schur_refinement_acceptances"]:
                self.assertIsInstance(result[field], int)
                self.assertGreaterEqual(result[field], 0)
                if value != "normal_refined":
                    self.assertEqual(result[field], 0)
            self.assertLessEqual(result["ipm_schur_refinement_acceptances"], result["ipm_schur_refinement_steps"])
        for arguments in [("--backend", "highs", "--ipm-linear-solver", "qdldl"),
                          ("--ipm-linear-solver", "sparselu", "--backend", "pdlp"),
                          ("--backend", "ipm", "--ipm-linear-solver", "invalid"),
                          ("--backend", "ipm", "--ipm-linear-solver", "sparselu",
                           "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_augmented"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_multifrontal"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_extended"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_refined"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_refined",
                           "--ipm-singleton-elimination", "on", "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_extended",
                           "--ipm-singleton-elimination", "on", "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_multifrontal",
                           "--ipm-singleton-elimination", "on", "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal_augmented",
                           "--ipm-singleton-elimination", "on", "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "normal",
                           "--ipm-singleton-elimination", "on",
                           "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "factorhighs",
                           "--ipm-dynamic-regularization", "on"),
                          ("--backend", "ipm", "--ipm-linear-solver", "qdldl_extended",
                           "--ipm-dynamic-regularization", "on")]:
            process, result = self.cli("solve", self.minimum, *arguments)
            self.assertEqual(process.returncode, 2, result)

    def test_maximum_retains_original_sense(self):
        process, report = self.cli("solve", self.maximum)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["model"]["sense"], "maximize")
        self.assertAlmostEqual(report["objective"], 7)
        if report["best_bound"] is not None:
            self.assertGreaterEqual(report["best_bound"] + 1e-7, report["objective"])

    def test_integer_solution(self):
        process, report = self.cli("solve", self.integer)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["status"], "optimal")
        self.assertTrue(report["model"]["is_mip"])
        self.assertAlmostEqual(report["objective"], 6)
        self.assertLessEqual(report["verification"]["max_integrality_violation"], 1e-6)

    def test_infeasible_is_not_primal_verified(self):
        process, report = self.cli("solve", self.infeasible)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["status"], "infeasible")
        self.assertFalse(report["has_solution"])
        self.assertIsNone(report["solution"])
        self.assertIsNone(report["objective"])
        self.assertFalse(report["verification"]["feasible"])

    def test_unbounded_uses_null_for_nonfinite_bound(self):
        process, report = self.cli("solve", self.unbounded)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["status"], "unbounded")
        self.assertIsNone(report["best_bound"])

    def test_errors_remain_structured(self):
        for arguments in [[], ["unknown"], ["solve", self.root / "missing.lp"],
                          ["solve", self.minimum, "--unknown", "1"],
                          ["solve", self.minimum, "--threads"],
                          ["solve", self.minimum, "--seed", "1", "--seed", "2"]]:
            with self.subTest(arguments=arguments):
                process, report = self.cli(*arguments)
                self.assertEqual(process.returncode, 2, report)
                self.assertEqual(report["status"], "error")
                self.assertTrue(report["message"])

    def test_error_message_escapes_json_control_characters(self):
        option = '--unknown"\\\t\n'
        process, report = self.cli("solve", self.minimum, option, "value")
        self.assertEqual(process.returncode, 2)
        self.assertIn(option, report["message"])

    def test_invalid_numeric_options(self):
        invalid = {"--threads": ["0", "-1", "1.5", "2147483648", "1x"],
                   "--seed": ["-1", "1.5", "2147483648", "no"],
                   "--time-limit": ["-1", "nan", "inf", "1e999", "1x"],
                   "--mip-gap": ["-1", "nan", "inf"]}
        for option, values in invalid.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    process, report = self.cli("solve", self.minimum, option, value)
                    self.assertEqual(process.returncode, 2, report)

    def test_zero_budget_is_incomplete(self):
        process, report = self.cli("solve", self.minimum, "--time-limit", "0")
        self.assertEqual(process.returncode, 4, report)
        self.assertEqual(report["status"], "time_limit")
        self.assertFalse(report["has_solution"])

    def test_pdlp_solves_lp_and_preserves_maximum_offset(self):
        for model, sense, objective in [(self.minimum, "minimize", 8 / 3),
                                         (self.maximum_offset, "maximize", 12)]:
            with self.subTest(model=model):
                process, report = self.cli("solve", model, "--backend", "pdlp")
                self.assertEqual(process.returncode, 0, report)
                self.assertEqual(report["backend"], "pdlp")
                self.assertEqual(report["options"]["backend"], "pdlp")
                self.assertEqual(report["status"], "optimal")
                self.assertEqual(report["model"]["sense"], sense)
                self.assertAlmostEqual(report["objective"], objective, delta=1e-6)
                self.assertTrue(report["verification"]["feasible"])
                self.assertGreater(report["pdlp_iterations"], 0)
                self.assertEqual(len(report["row_dual"]), report["model"]["rows"])
                for field in ["valid_input", "dual_feasible", "gap_satisfied", "numerically_optimal"]:
                    self.assertIs(report["lp_optimality"][field], True)
                self.assertEqual(self.benchmark.classify(process.returncode, report), "resolved")

    def test_pdlp_rejects_milp_and_multiple_threads(self):
        process, report = self.cli("solve", self.integer, "--backend", "pdlp")
        self.assertEqual(process.returncode, 2, report)
        self.assertEqual(report["backend"], "pdlp")
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["has_solution"])
        process, report = self.cli("solve", self.minimum, "--backend", "pdlp", "--threads", 2)
        self.assertEqual(process.returncode, 2, report)
        self.assertEqual(report["status"], "error")

    def test_pdlp_zero_budgets_are_incomplete(self):
        for flag, status in [("--iteration-limit", "iteration_limit"), ("--time-limit", "time_limit")]:
            with self.subTest(flag=flag):
                process, report = self.cli("solve", self.minimum, "--backend", "pdlp", flag, 0)
                self.assertEqual(process.returncode, 4, report)
                self.assertEqual(report["backend"], "pdlp")
                self.assertEqual(report["status"], status)
                self.assertEqual(report["pdlp_iterations"], 0)

    def test_ipm_solves_lp_and_preserves_maximum_offset(self):
        for model, objective in [(self.minimum, 8 / 3), (self.maximum_offset, 12)]:
            with self.subTest(model=model):
                process, report = self.cli("solve", model, "--backend", "ipm")
                self.assertEqual(process.returncode, 0, report)
                self.assertEqual(report["backend"], "ipm")
                self.assertEqual(report["status"], "optimal")
                self.assertAlmostEqual(report["objective"], objective, delta=1e-6)
                self.assertIsNone(report["options"]["pdlp"])
                self.assertEqual(report["options"]["ipm"]["max_iterations"], 200)
                self.assertIsNone(report["pdlp_iterations"])
                self.assertTrue(self.benchmark.valid_lp_optimality(report))
                self.assertEqual(self.benchmark.classify(process.returncode, report), "resolved")
                for field in ["ipm_iterations", "ipm_factorizations", "ipm_refinement_steps"]:
                    self.assertIs(type(report[field]), int)
                    self.assertGreaterEqual(report[field], 0)

    def test_ipm_budget_options_are_independent_of_argument_order(self):
        for flags in [["--iteration-limit", "0", "--scaling-passes", "0", "--backend", "ipm"],
                      ["--backend", "ipm", "--iteration-limit", "0", "--scaling-passes", "0"],
                      ["--iteration-limit", "2147483648", "--backend", "ipm", "--time-limit", "0"]]:
            with self.subTest(flags=flags):
                process, report = self.cli("solve", self.minimum, *flags)
                self.assertEqual(process.returncode, 4, report)
                self.assertEqual(report["ipm_iterations"], 0)
                self.assertEqual(report["options"]["ipm"]["max_iterations"], int(flags[flags.index("--iteration-limit") + 1]))
                self.assertEqual(report["status"], "time_limit" if "--time-limit" in flags else "iteration_limit")
                if "--scaling-passes" in flags:
                    self.assertEqual(report["options"]["ipm"]["scaling_passes"], 0)

    def test_ipm_rejects_milp_and_pdlp_specific_options(self):
        process, report = self.cli("solve", self.integer, "--backend", "ipm")
        self.assertEqual(process.returncode, 2, report)
        self.assertFalse(report["has_solution"])
        for flags in [["--pdlp-mode", "adaptive"], ["--pdlp-history-size", "0"], ["--threads", "2"],
                      ["--iteration-limit", "-1"], ["--iteration-limit", "9223372036854775808"],
                      ["--scaling-passes", "21"]]:
            with self.subTest(flags=flags):
                process, report = self.cli("solve", self.minimum, "--backend", "ipm", *flags)
                self.assertEqual(process.returncode, 2, report)

    def test_backend_parameter_validation_and_int64_iteration_limit(self):
        invalid = [["--backend", "unknown"], ["--iteration-limit", "100"], ["--scaling-passes", "0"],
                   ["--pdlp-mode", "adaptive"], ["--backend", "pdlp", "--pdlp-mode", "unknown"],
                   ["--pdlp-history-size", "8"],
                   ["--backend", "pdlp", "--pdlp-history-size", "-1"],
                   ["--backend", "pdlp", "--pdlp-history-size", "33"],
                   ["--backend", "pdlp", "--pdlp-history-size", "1.5"],
                   ["--backend", "pdlp", "--iteration-limit", "-1"],
                   ["--backend", "pdlp", "--iteration-limit", "1.5"],
                   ["--backend", "pdlp", "--iteration-limit", "9223372036854775808"],
                   ["--backend", "pdlp", "--scaling-passes", "-1"],
                   ["--backend", "pdlp", "--scaling-passes", "21"]]
        for flags in invalid:
            with self.subTest(flags=flags):
                process, report = self.cli("solve", self.minimum, *flags)
                self.assertEqual(process.returncode, 2, report)
                self.assertEqual(report["status"], "error")
        process, report = self.cli("solve", self.minimum, "--iteration-limit", "2147483648",
                                   "--scaling-passes", "0", "--backend", "pdlp", "--time-limit", "0")
        self.assertEqual(process.returncode, 4, report)
        self.assertEqual(report["options"]["pdlp"]["max_iterations"], 2147483648)
        self.assertEqual(report["options"]["pdlp"]["scaling_passes"], 0)

    def test_highs_lp_report_has_common_optimality_diagnostics(self):
        process, report = self.cli("solve", self.minimum, "--backend", "highs")
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["backend"], "highs")
        self.assertIsNone(report["options"]["pdlp"])
        self.assertIsNone(report["options"]["ipm"])
        self.assertIsNone(report["pdlp_iterations"])
        for field in ["pdlp_acceleration_attempts", "pdlp_acceleration_accepted", "pdlp_operator_passes",
                      "ipm_iterations", "ipm_factorizations", "ipm_refinement_steps"]:
            self.assertIsNone(report[field])
        self.assertTrue(self.benchmark.valid_lp_optimality(report))
        self.assertEqual(len(report["row_dual"]), report["model"]["rows"])

    def test_pdlp_modes_are_explicit_and_recorded(self):
        for mode in ["baseline", "adaptive", "subspace", "halpern", "halpern_reflected"]:
            with self.subTest(mode=mode):
                process, report = self.cli("solve", self.minimum, "--backend", "pdlp", "--pdlp-mode", mode)
                self.assertEqual(process.returncode, 0, report)
                self.assertEqual(report["options"]["pdlp"]["mode"], mode)
                self.assertEqual(report["status"], "optimal")

    def test_pdlp_history_budget_and_work_counters_are_recorded(self):
        for size in [0, 32]:
            with self.subTest(size=size):
                process, report = self.cli("solve", self.minimum, "--backend", "pdlp",
                                           "--pdlp-mode", "subspace", "--pdlp-history-size", size)
                self.assertEqual(process.returncode, 0, report)
                self.assertEqual(report["options"]["pdlp"]["history_size"], size)
                for field in ["pdlp_acceleration_attempts", "pdlp_acceleration_accepted", "pdlp_operator_passes"]:
                    self.assertIs(type(report[field]), int)
                    self.assertGreaterEqual(report[field], 0)
                self.assertLessEqual(report["pdlp_acceleration_accepted"], report["pdlp_acceleration_attempts"])
                if size == 0:
                    self.assertEqual(report["pdlp_acceleration_attempts"], 0)

    def test_direct_parser_diagnostics_do_not_pollute_json(self):
        diagnostic_model = self.root / "parser diagnostic.lp"
        diagnostic_model.write_text("Minimize\n obj: x\nSubject To\n HiGHS_Rrow: x >= 1\nEnd\n", encoding="utf-8")
        process, report = self.cli("solve", diagnostic_model)
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report["status"], "optimal")
        unsupported_model = self.root / "unsupported indicator.lp"
        unsupported_model.write_text("Minimize\n obj: x\nSubject To\n indicator: z = 1 -> x >= 1\nBinary\n z\nEnd\n", encoding="utf-8")
        process, report = self.cli("solve", unsupported_model)
        self.assertEqual(process.returncode, 2, report)
        self.assertEqual(report["status"], "error")

    def test_native_filesystem_error_remains_utf8(self):
        if sys.platform != "win32":
            self.skipTest("此项覆盖 Windows 原生错误消息")
        process, report = self.cli("solve", self.minimum, "--output", self.root / "invalid?name.json")
        self.assertEqual(process.returncode, 2, report)
        self.assertTrue(report["message"])

    def test_report_file_and_overwrite_protection(self):
        destination = self.root / "saved report.json"
        process, report = self.cli("solve", self.minimum, "--output", destination)
        self.assertEqual(process.returncode, 0, report)
        original = destination.read_bytes()
        self.assertEqual(parse_json(original), report)
        process, error = self.cli("solve", self.maximum, "--output", destination)
        self.assertEqual(process.returncode, 2, error)
        self.assertEqual(destination.read_bytes(), original)

    def test_benchmark_success_and_failure_are_persisted(self):
        output = self.root / "benchmark mixed"
        process = self.bench(self.minimum, self.infeasible, self.unbounded,
                             self.root / "missing model.lp", "--solver", SOLVER,
                             "--output-dir", output, "--time-limit", "2", "--seeds", "0", "7")
        self.assertEqual(process.returncode, 1, process.stderr)
        summary = parse_json(process.stdout)
        self.assertEqual(summary["run_count"], 8)
        self.assertEqual(summary["optimal_count"], 2)
        self.assertEqual(summary["infeasible_count"], 2)
        self.assertEqual(summary["unbounded_count"], 2)
        self.assertEqual(summary["resolved_count"], 6)
        self.assertEqual(summary["unresolved_count"], 2)
        records = [parse_json(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 8)
        self.assertEqual(records[-1]["returncode"], 2)
        self.assertEqual(records[-1]["classification"], "solver_error")
        self.assertEqual(records[-1]["penalized_capped_seconds"], 2)
        self.assertIsNotNone(records[0]["model_sha256"])
        expected = math.exp(sum(math.log(record["penalized_capped_seconds"] + 10) for record in records) / 8) - 10
        self.assertAlmostEqual(summary["penalized_capped_wall_sgm10_seconds"], expected)
        repeat = self.bench(self.minimum, "--solver", SOLVER, "--output-dir", output)
        self.assertEqual(repeat.returncode, 2)
        self.assertEqual(len((output / "runs.jsonl").read_text(encoding="utf-8").splitlines()), 8)

    def test_benchmark_missing_executable_is_persisted(self):
        output = self.root / "benchmark missing executable"
        process = self.bench(self.minimum, "--solver", self.root / "missing.exe", "--output-dir", output)
        self.assertEqual(process.returncode, 1, process.stderr)
        record = parse_json((output / "runs.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(record["classification"], "process_error")
        self.assertIsNone(record["returncode"])
        self.assertTrue(record["error"])

    def test_benchmark_records_both_backends(self):
        for backend in ["highs", "pdlp", "ipm"]:
            with self.subTest(backend=backend):
                output = self.root / ("benchmark comparison " + backend)
                flags = ["--iteration-limit", "200000", "--scaling-passes", "5", "--pdlp-mode", "subspace", "--pdlp-history-size", "0"] if backend == "pdlp" else []
                if backend == "ipm":
                    flags = ["--iteration-limit", "200", "--scaling-passes", "5"]
                process = self.bench(self.minimum, self.maximum_offset, "--solver", SOLVER,
                                     "--backend", backend, "--time-limit", "5", "--output-dir", output, *flags)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                self.assertEqual(parse_json(process.stdout)["optimal_count"], 2)
                manifest = parse_json((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["backend"], backend)
                records = [parse_json(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
                for record in records:
                    self.assertEqual(record["report"]["backend"], backend)
                    if backend == "highs":
                        self.assertNotIn("--iteration-limit", record["command"])
                        self.assertNotIn("--scaling-passes", record["command"])
                        self.assertNotIn("--pdlp-mode", record["command"])
                        self.assertNotIn("--pdlp-history-size", record["command"])
                    elif backend == "pdlp":
                        self.assertEqual(record["report"]["options"]["pdlp"]["mode"], "subspace")
                        self.assertEqual(record["report"]["options"]["pdlp"]["history_size"], 0)
                        self.assertEqual(manifest["pdlp_requested_options"]["history_size"], 0)
                    else:
                        self.assertNotIn("--pdlp-mode", record["command"])
                        self.assertNotIn("--pdlp-history-size", record["command"])
                        self.assertIsNone(manifest["pdlp_requested_options"])
                        self.assertEqual(manifest["ipm_requested_options"], {"max_iterations": 200, "scaling_passes": 5,
                                                                            "eliminate_singletons": None,
                                                                            "dynamic_regularization": None,
                                                                            "equilibrate_kkt": None,
                                                                            "linear_accuracy": None,
                                                                            "central_corrections": None,
                                                                            "direction_retries": None,
                                                                            "krylov_history_size": None,
                                                                            "linear_solver": None,
                                                                            "inexact_prefilter": None, "eliminate_fixed_variables": None,
                                                                            "presolve_singleton_rows": None,
                                                                            "eliminate_presolved_fixed_variables": None})
                        self.assertEqual(record["report"]["options"]["ipm"]["max_iterations"], 200)

    def test_benchmark_pdlp_rejections_and_limits_are_persisted(self):
        output = self.root / "benchmark pdlp incomplete"
        process = self.bench(self.minimum, self.integer, "--solver", SOLVER, "--backend", "pdlp",
                             "--iteration-limit", "0", "--time-limit", "2", "--output-dir", output)
        self.assertEqual(process.returncode, 1, process.stderr)
        summary = parse_json(process.stdout)
        self.assertEqual(summary["resolved_count"], 0)
        self.assertEqual(summary["classification_counts"], {"incomplete": 1, "solver_error": 1})
        records = [parse_json(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([record["report"]["backend"] for record in records], ["pdlp", "pdlp"])
        self.assertAlmostEqual(summary["penalized_capped_wall_sgm10_seconds"], 2)

    def test_benchmark_timeout_is_persisted(self):
        output = self.root / "benchmark timeout"
        arguments = [str(BENCHMARK), str(self.minimum), "--solver", str(SOLVER),
                     "--output-dir", str(output), "--time-limit", "1", "--wall-timeout", "1"]
        timeout = subprocess.TimeoutExpired([str(SOLVER)], timeout=1, output=b"partial", stderr=b"failure")
        with mock.patch.object(sys, "argv", arguments), \
                mock.patch.object(self.benchmark.subprocess, "run", side_effect=timeout), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.benchmark.main(), 1)
        record = parse_json((output / "runs.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(record["classification"], "wall_timeout")
        self.assertEqual(record["stdout"], "partial")
        self.assertEqual(record["stderr"], "failure")
        self.assertEqual(record["penalized_capped_seconds"], 1)

    def test_benchmark_handles_native_stderr_but_rejects_bad_stdout_encoding(self):
        args = self.benchmark.argparse.Namespace(solver=SOLVER, time_limit=1, threads=1,
                                                mip_gap=1e-4, wall_timeout=2, backend="highs",
                                                iteration_limit=None, scaling_passes=None)
        valid_report = b'{"schema_version":1,"status":"infeasible","backend":"highs"}'
        process = subprocess.CompletedProcess([], 0, stdout=valid_report, stderr=b"native: \xb2\xe2")
        with mock.patch.object(self.benchmark.subprocess, "run", return_value=process):
            record = self.benchmark.run_one(args, self.minimum, 0, 0)
        self.assertEqual(record["classification"], "resolved")
        self.assertIn("\ufffd", record["stderr"])
        process.stdout = b'{"schema_version":1,"status":"infeasible","message":"\xb2"}'
        with mock.patch.object(self.benchmark.subprocess, "run", return_value=process):
            record = self.benchmark.run_one(args, self.minimum, 0, 1)
        self.assertEqual(record["classification"], "invalid_json")
        self.assertEqual(record["penalized_capped_seconds"], 1)
        args.backend = "pdlp"
        process.stdout = valid_report
        with mock.patch.object(self.benchmark.subprocess, "run", return_value=process):
            record = self.benchmark.run_one(args, self.minimum, 0, 2)
        self.assertEqual(record["classification"], "backend_mismatch")
        self.assertEqual(record["penalized_capped_seconds"], 1)

    def test_benchmark_rejects_nonfinite_and_false_optimal_reports(self):
        for value in ['{"objective":NaN}', '{"objective":Infinity}', '{"objective":1e999}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.benchmark.strict_json(value)
        self.assertEqual(self.benchmark.classify(0, {"schema_version": 1, "status": "optimal"}), "invalid_report")
        self.assertEqual(self.benchmark.classify(0, {"schema_version": 1, "status": "time_limit"}), "exit_status_mismatch")
        candidate = {"schema_version": 1, "status": "optimal", "backend": "highs", "has_solution": True,
                     "solution": [1.0], "model": {"columns": 1}, "objective": 1.0,
                     "verification": {"valid_input": True, "feasible": True,
                                      "objective_consistent": True, "recomputed_objective": 1.0}}
        self.assertEqual(self.benchmark.classify(0, candidate), "resolved")
        for objective in [None, True, "1.0", float("nan"), float("inf"), 10 ** 1000]:
            with self.subTest(objective=objective):
                self.assertEqual(self.benchmark.classify(0, dict(candidate, objective=objective)), "invalid_report")
        for solution in [None, [], [None], [True], [float("inf")]]:
            with self.subTest(solution=solution):
                self.assertEqual(self.benchmark.classify(0, dict(candidate, solution=solution)), "invalid_report")
        incomplete = dict(candidate)
        del incomplete["objective"]
        self.assertEqual(self.benchmark.classify(0, incomplete), "invalid_report")

    def test_benchmark_requires_pdlp_numerical_optimality_evidence(self):
        candidate = {"schema_version": 1, "backend": "pdlp", "status": "optimal", "has_solution": True,
                     "objective": 1.0, "best_bound": None, "solution": [1.0], "row_dual": [1.0],
                     "model": {"columns": 1, "rows": 1},
                     "verification": {"valid_input": True, "feasible": True,
                                      "objective_consistent": True, "recomputed_objective": 1.0},
                     "lp_optimality": {"valid_input": True, "dual_feasible": True, "gap_satisfied": True,
                                       "numerically_optimal": True, "max_dual_residual": 0.0,
                                       "dual_objective_estimate": 1.0, "objective_gap": 0.0,
                                       "relative_objective_gap": 0.0, "corrected_dual_bound": None}}
        self.assertEqual(self.benchmark.classify(0, candidate), "resolved")
        self.assertEqual(self.benchmark.classify(0, dict(candidate, lp_optimality=None)), "invalid_report")
        for field in ["valid_input", "dual_feasible", "gap_satisfied", "numerically_optimal"]:
            for value in [None, False, 1]:
                with self.subTest(field=field, value=value):
                    altered = dict(candidate["lp_optimality"], **{field: value})
                    self.assertEqual(self.benchmark.classify(0, dict(candidate, lp_optimality=altered)), "invalid_report")
        without_bound = dict(candidate)
        del without_bound["best_bound"]
        self.assertEqual(self.benchmark.classify(0, without_bound), "resolved")


if __name__ == "__main__":
    unittest.main()
