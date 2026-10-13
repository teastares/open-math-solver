#!/usr/bin/env python3
"""用已知 KKT 点交叉检查 HiGHS、PDLP 和自主内点法，不向求解器提供答案。"""

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest


SOLVER = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], *sys.argv[2:]]
BACKENDS = ("highs", "pdlp", "ipm")
INF = math.inf


def expression(coefficients):
    return " ".join(f"{value:+.17g} x{index}" for index, value in enumerate(coefficients))


def planted_problem(seed, maximize, rescale):
    generator = random.Random(seed)
    columns, rows = 6, 4
    optimum = [generator.choice([-2.0, -1.0, 0.5, 1.5, 2.0]) for _ in range(columns)]
    matrix = [[float(generator.randint(-2, 2)) for _ in range(columns)] for _ in range(rows)]
    dual = [1.5, -0.75, 0.5, 0.0]
    reduced = [1.0, -2.0, 0.0, 0.5, -0.25, 0.0]
    costs = [math.fsum(matrix[i][j] * dual[i] for i in range(rows)) + reduced[j]
             for j in range(columns)]
    lower = [value if reduced[j] > 0 else value - 3.0 for j, value in enumerate(optimum)]
    upper = [value if reduced[j] < 0 else value + 3.0 for j, value in enumerate(optimum)]
    sense = -1.0 if maximize else 1.0
    offset = 7.25
    expected = sense * math.fsum(a * b for a, b in zip(costs, optimum)) + offset
    lines = ["Maximize" if maximize else "Minimize",
             " obj: " + expression([sense * c for c in costs]) + f" + {offset}", "Subject To"]
    factors = [0.01, 10.0, 1.0, 0.1] if rescale else [1.0] * rows
    for i, row in enumerate(matrix):
        coefficients = [a * factors[i] for a in row]
        activity = math.fsum(a * b for a, b in zip(coefficients, optimum))
        relation = ">=" if i == 0 else "<=" if i == 1 else "="
        if i < 3:
            lines.append(f" r{i}: {expression(coefficients)} {relation} {activity:.17g}")
        else:
            # 非活跃的双边约束拆为两行，保持零对偶乘子和同一已知最优点。
            lines.append(f" r{i}l: {expression(coefficients)} >= {activity - factors[i]:.17g}")
            lines.append(f" r{i}u: {expression(coefficients)} <= {activity + factors[i]:.17g}")
    lines.append("Bounds")
    lines.extend(f" {low:.17g} <= x{j} <= {high:.17g}"
                 for j, (low, high) in enumerate(zip(lower, upper)))
    lines.append("End")
    return "\n".join(lines) + "\n", expected


@dataclass(frozen=True)
class KktProblem:
    name: str
    matrix: tuple
    row_bounds: tuple
    column_bounds: tuple
    optimum: tuple
    dual: tuple
    reduced: tuple
    offset: float = 0.0
    maximize: bool = False

    @property
    def costs(self):
        sign = -1 if self.maximize else 1
        return tuple(sign * math.fsum([self.reduced[j], *(
            row[j] * multiplier for row, multiplier in zip(self.matrix, self.dual))])
            for j in range(len(self.optimum)))

    @property
    def variable_objective(self):
        return float(sum(Fraction(cost) * Fraction(value)
                         for cost, value in zip(self.costs, self.optimum)))

    @property
    def objective(self):
        return self.offset + self.variable_objective

    def mps(self):
        # G 行配合 RANGES 表示真正双边行，不拆成两个单边约束。
        lines = [f"NAME {self.name}", "OBJSENSE", " MAX" if self.maximize else " MIN",
                 "ROWS", " N OBJ"]
        rhs, ranges = [], []
        for i, (lower, upper) in enumerate(self.row_bounds):
            if lower == upper:
                kind, value = "E", lower
            elif math.isfinite(lower):
                kind, value = "G", lower
                if math.isfinite(upper):
                    ranges.append((i, upper - lower))
            else:
                kind, value = "L", upper
            lines.append(f" {kind} R{i}")
            rhs.append((i, value))
        lines.append("COLUMNS")
        for j, cost in enumerate(self.costs):
            lines.append(f"    X{j} OBJ {cost:.17g}")
            lines.extend(f"    X{j} R{i} {row[j]:.17g}"
                         for i, row in enumerate(self.matrix) if row[j] != 0)
        # MPS 的目标行 RHS 为目标常数的相反数。
        lines.extend(["RHS", f"    RHS1 OBJ {-self.offset:.17g}"])
        lines.extend(f"    RHS1 R{i} {value:.17g}" for i, value in rhs)
        if ranges:
            lines.append("RANGES")
            lines.extend(f"    RNG1 R{i} {value:.17g}" for i, value in ranges)
        lines.append("BOUNDS")
        for j, (lower, upper) in enumerate(self.column_bounds):
            if lower == upper:
                lines.append(f" FX BND1 X{j} {lower:.17g}")
            elif lower == -INF and upper == INF:
                lines.append(f" FR BND1 X{j}")
            else:
                lines.append(f" MI BND1 X{j}" if lower == -INF
                             else f" LO BND1 X{j} {lower:.17g}")
                if upper != INF:
                    lines.append(f" UP BND1 X{j} {upper:.17g}")
        return "\n".join([*lines, "ENDATA", ""])


def constructed_problems():
    mixed = KktProblem(
        "mixed_sides", ((1, 1, 0, 0), (0, 1, 1, 0), (1, 0, 1, -1), (0, 0, 1, 1)),
        ((-1, 3), (-6, -2), (-1, -1), (3, 8)),
        ((1, 5), (-INF, INF), (1, 1), (-2, 4)),
        (2, -3, 1, 4), (1.5, -0.5, 0.25, 0), (0, 0, 2, -1), 17.25)
    yield mixed
    yield KktProblem(
        "free_optimal_face", ((1, 2, -1), (-2, 1, 3)), ((3, 3), (10, 10)),
        ((-INF, INF),) * 3, (-2, 3, 1), (0.5, -1.25), (0, 0, 0), -8.5, True)
    yield KktProblem(
        "dependent_equalities",
        ((1, 2, 0), (1, 2, 0), (-2, -4, 0), (0, 1, 1), (1, 3, 1)),
        ((4, 4), (4, 4), (-8, -8), (0, 0), (4, 4)),
        ((-5, 5), (-2, 4), (-3, 3)), (2, 1, -1), (1, -0.5, 0.25, -1, 0.75),
        (0, 0, 0), 0.125)
    yield KktProblem(
        "no_strict_interior", ((1, 1, 0), (-1, -1, 0), (0, 1, 1)),
        ((3, INF), (-3, INF), (-1, -1)), ((1, 5), (-4, 2), (-INF, INF)),
        (1, 2, -3), (2, 0, -0.5), (0, 0, 0), -2.25)
    yield KktProblem(
        "fixed_columns", ((1, -2, 1, 0), (0, 1, 1, 1), (2, 0, -1, 1)),
        ((6, 10), (6, 6), (3, 3)), ((2, 2), (-1, -1), (0, 8), (-INF, INF)),
        (2, -1, 4, 3), (0, 1.25, -0.5), (3, -2, 0, 0), 9.5, True)
    yield KktProblem(
        "one_sided_columns", ((1, 1, 1), (0, 1, -1)), ((2, 2), (0, 4)),
        ((-INF, -2), (3, INF), (-INF, INF)),
        (-2, 3, 1), (0.5, 0), (-1, 2, 0), -3, True)
    # 正对角变换保持最优目标，同时覆盖行、列量级不一致的同一数学问题。
    row_scale, column_scale = (1 / 64, 64, 1 / 8, 8), (1 / 64, 8, 1 / 4, 16)
    yield KktProblem(
        "mixed_scales",
        tuple(tuple(value * row_scale[i] / column_scale[j] for j, value in enumerate(row))
              for i, row in enumerate(mixed.matrix)),
        tuple((lower * scale, upper * scale)
              for (lower, upper), scale in zip(mixed.row_bounds, row_scale)),
        tuple((lower * scale, upper * scale)
              for (lower, upper), scale in zip(mixed.column_bounds, column_scale)),
        tuple(value * scale for value, scale in zip(mixed.optimum, column_scale)),
        tuple(value / scale for value, scale in zip(mixed.dual, row_scale)),
        tuple(value / scale for value, scale in zip(mixed.reduced, column_scale)), mixed.offset)
    yield KktProblem(
        "large_objective_offset", ((1, 1, 1), (1, -1, 0)), ((4, 4), (0, 4)),
        ((0, 2), (-1, 4), (0, 5)), (2, -1, 3), (0.5, 0), (-2, 1, 0), 2**40, True)


class ConstructedKktTests(unittest.TestCase):
    def test_exact_primal_dual_certificates(self):
        problems = list(constructed_problems())
        self.assertEqual(len(problems), 8)
        self.assertEqual(len({problem.name for problem in problems}), 8)
        for problem in problems:
            with self.subTest(problem=problem.name):
                x = tuple(map(Fraction, problem.optimum))
                y = tuple(map(Fraction, problem.dual))
                reduced = tuple(map(Fraction, problem.reduced))
                activity = tuple(sum(Fraction(a) * value for a, value in zip(row, x))
                                 for row in problem.matrix)
                self.assertEqual(len(x), len(problem.column_bounds))
                self.assertEqual(len(y), len(activity))
                self.assertEqual(len(y), len(problem.row_bounds))
                self.assertEqual(len(reduced), len(x))
                self.assertTrue(all(len(row) == len(x) for row in problem.matrix))
                dual_objective = Fraction(0)
                for values, multipliers, bounds in ((x, reduced, problem.column_bounds),
                                                    (activity, y, problem.row_bounds)):
                    for value, multiplier, (lower, upper) in zip(values, multipliers, bounds):
                        self.assertGreaterEqual(value, lower)
                        self.assertLessEqual(value, upper)
                        if multiplier:
                            endpoint = lower if multiplier > 0 else upper
                            self.assertTrue(math.isfinite(endpoint))
                            self.assertEqual(value, Fraction(endpoint))
                            dual_objective += multiplier * Fraction(endpoint)
                sign = -1 if problem.maximize else 1
                for j, cost in enumerate(problem.costs):
                    self.assertEqual(sign * Fraction(cost), reduced[j] + sum(
                        Fraction(row[j]) * multiplier for row, multiplier in zip(problem.matrix, y)))
                self.assertEqual(sign * Fraction(problem.variable_objective), dual_objective)


class PdlpComparison(unittest.TestCase):
    def solve_and_check(self, model, backend, expected, maximize, require_bound=False,
                        variable_scale=None):
        process = subprocess.run(
            [str(SOLVER), "solve", str(model), "--backend", backend, "--time-limit", "5"],
            capture_output=True, timeout=15, check=False)
        report = json.loads(process.stdout.decode("utf-8", errors="strict"))
        self.assertEqual(process.returncode, 0, (backend, report))
        self.assertEqual(report["status"], "optimal", (backend, report))
        self.assertEqual(report["backend"], backend)
        self.assertIs(report["has_solution"], True)
        for field in ("valid_input", "feasible", "objective_consistent"):
            self.assertIs(report["verification"][field], True, (backend, report))
        for field in ("valid_input", "dual_feasible", "gap_satisfied", "numerically_optimal"):
            self.assertIs(report["lp_optimality"][field], True, (backend, report))
        scale = max(1.0, abs(expected)) if variable_scale is None else variable_scale
        self.assertLessEqual(abs(report["objective"] - expected),
                             max(2e-6 * scale, 2 * math.ulp(expected)), (backend, report))
        if backend in ("pdlp", "ipm"):
            self.assertEqual(report["best_bound"], report["lp_optimality"]["corrected_dual_bound"])
        if require_bound:
            self.assertIsNotNone(report["best_bound"])
        if report["best_bound"] is not None:
            directed_gap = ((report["best_bound"] - expected) if maximize
                            else (expected - report["best_bound"]))
            self.assertGreaterEqual(directed_gap, -max(1e-7 * scale, 2 * math.ulp(expected)),
                                    (backend, report))
        return report

    def test_planted_optima_match_all_engines(self):
        with tempfile.TemporaryDirectory(prefix="solver-pdlp-comparison-") as directory:
            for seed in (7, 19, 41):
                for maximize in (False, True):
                    for rescale in (False, True):
                        with self.subTest(seed=seed, maximize=maximize, rescale=rescale):
                            contents, expected = planted_problem(seed, maximize, rescale)
                            model = Path(directory) / f"model-{seed}-{int(maximize)}-{int(rescale)}.lp"
                            model.write_text(contents, encoding="utf-8")
                            for backend in BACKENDS:
                                with self.subTest(backend=backend):
                                    self.solve_and_check(model, backend, expected, maximize,
                                                         require_bound=True)

    def test_constructed_domains_degeneracy_and_scales(self):
        with tempfile.TemporaryDirectory(prefix="solver-lp-kkt-") as directory:
            for problem in constructed_problems():
                model = Path(directory) / f"{problem.name}.mps"
                model.write_text(problem.mps(), encoding="utf-8")
                for backend in BACKENDS:
                    with self.subTest(problem=problem.name, backend=backend):
                        scale = max(1.0, abs(problem.variable_objective))
                        report = self.solve_and_check(
                            model, backend, problem.objective, problem.maximize, variable_scale=scale)
                        self.assertEqual(report["model"]["objective_offset"], problem.offset)
                        self.assertEqual(report["model"]["sense"],
                                         "maximize" if problem.maximize else "minimize")
                        self.assertEqual(report["model"]["columns"], len(problem.optimum))
                        self.assertEqual(report["model"]["rows"], len(problem.matrix))
                        self.assertEqual(len(report["solution"]), len(problem.optimum))
                        self.assertEqual(len(report["row_dual"]), len(problem.matrix))
                        self.assertTrue(all(math.isfinite(value) for value in report["solution"]))
                        self.assertTrue(all(math.isfinite(value) for value in report["row_dual"]))
                        # 独立检查不含常数的目标，禁止大 offset 掩盖错误最优点。
                        variable_objective = math.fsum(cost * value for cost, value in
                                                       zip(problem.costs, report["solution"]))
                        self.assertLessEqual(abs(variable_objective - problem.variable_objective),
                                             2e-6 * scale, report)


if __name__ == "__main__":
    unittest.main()
