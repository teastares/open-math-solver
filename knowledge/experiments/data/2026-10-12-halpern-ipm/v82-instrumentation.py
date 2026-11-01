"""v82 阶段诊断：复用锁定的 v47 计时器，区分冻结 v66 的矩形数值路径。"""

import hashlib
import json
from pathlib import Path
import re
import types


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
HELPER = DATA / "v47-profile-build.py"
PLAN = DATA / "v82-plan.json"
HELPER_SHA256 = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
# 主代理已冻结的事前计划身份。
PLAN_SHA256 = "b5df77438c3905566b0a8ddb02b84c2c2f65484facfe26bf559ec2ae085ec44b"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locked_inputs():
    """旧辅助器和事前计划按原始字节锁定；不产生导入缓存或构建产物。"""
    require(re.fullmatch(r"[0-9a-f]{64}", PLAN_SHA256) is not None,
            "v82 计划身份仍为待冻结占位")
    for path, expected in ((HELPER, HELPER_SHA256), (PLAN, PLAN_SHA256)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"锁定来源身份改变：{path.name}")
    return json.loads(PLAN.read_text(encoding="utf-8"))


_plan = locked_inputs()
_legacy = types.ModuleType("solver_v82_locked_v47")
_legacy.__file__ = str(HELPER)
exec(compile(HELPER.read_text(encoding="utf-8"), str(HELPER), "exec"), _legacy.__dict__)

STAGES = _legacy.STAGES + (
    "cone_multiply", "cone_transpose", "cholmod_symbolic", "cholmod_numeric",
    "cholmod_backsolve", "cholmod_values_copy",
    "normal_ordering", "aat_scaling", "cholmod_aat_numeric",
)
require(len(STAGES) == 35 and len(set(STAGES)) == len(STAGES), "阶段容量或名称重复")
require(list(STAGES) == _plan["profile_stages"], "阶段名称和顺序必须符合冻结计划")
require(_plan["profile_environment"] == _legacy.ENVIRONMENT and
        _plan["profile_prefix"] == _legacy.PREFIX, "旧诊断环境变量及输出前缀必须保持")


def replace_once(source, before, after, label):
    require(source.count(before) == 1, f"{label} 锚点不唯一")
    require(before != after, f"{label} 变换不得为空")
    return source.replace(before, after, 1)


_old_names = "inline constexpr const char* names[] = {" + ",".join(
    json.dumps(name) for name in _legacy.STAGES) + "};"
_new_names = "inline constexpr const char* names[] = {" + ",".join(
    json.dumps(name) for name in STAGES) + "};"
_old_capacity = f"inline thread_local std::array<Entry, {len(_legacy.STAGES)}> entries{{}};"
_new_capacity = f"inline thread_local std::array<Entry, {len(STAGES)}> entries{{}};"
HEADER = replace_once(_legacy.HEADER, _old_names, _new_names, "阶段名称")
HEADER = replace_once(HEADER, _old_capacity, _new_capacity, "固定容量")
# 保留旧头文件名、命名空间、环境变量、输出前缀和双翻译单元自测的全部字节。
SELF_MAIN = _legacy.SELF_MAIN
SELF_CROSS = _legacy.SELF_CROSS


def transform(source, unit):
    """沿用旧作用域、表达式物化与可逆记录；每次替换最终都核对唯一字面量。"""
    require(unit in ("ipm", "normal"), "未知翻译单元")
    source = source.replace("\r\n", "\n")
    require("solver_v47_profile::" not in source and
            '#include "v47-profile.hpp"' not in source, "拒绝重复插桩")
    original = source
    changes = []

    def replace(label, before, after):
        nonlocal source
        source = replace_once(source, before, after, f"{unit}/{label}")
        changes.append({"label": label, "before": before, "after": after})

    def scope(label, pattern, stage):
        matches = list(re.finditer(pattern, source))
        require(len(matches) == 1, f"{unit}/{label} 函数锚点数量为 {len(matches)}")
        before = matches[0].group()
        addition = f"\n  solver_v47_profile::Scope v47_scope({STAGES.index(stage)});"
        if stage == "total":
            addition = "\n  solver_v47_profile::Report v47_report;" + addition
        replace(label, before, before + addition)

    def expression(label, expression, stage, value_type=None):
        name = "call" if value_type is None else f"materialize<{value_type}>"
        replace(label, expression,
                f"solver_v47_profile::{name}({STAGES.index(stage)}, [&] {{ return {expression}; }})")

    namespace = "namespace solver {" if unit == "ipm" else "namespace solver::detail {"
    replace("shared_header", namespace, '#include "v47-profile.hpp"\n\n' + namespace)
    if unit == "ipm":
        for label, pattern, stage in (
            ("total", r"SolveResult solve_ipm\([^{}]+\) \{", "total"),
            ("canonicalize", r"ConeModel canonicalize\([^{}]+\) \{", "canonicalize"),
            ("kkt_setup", r"KktFactor\([^{}]+krylov_history_\([^{}]+\) \{", "kkt_setup"),
            ("factor", r"void factor\([^{}]+\) \{", "factor"),
            ("solve", r"std::vector<double> solve\([^{}]+\) const \{", "solve"),
            ("krylov", r"std::vector<double> krylov_refine\([^{}]+\) const \{", "krylov"),
            ("direct", r"std::vector<double> direct\([^{}]+\) const \{", "direct"),
            ("solve_factored", r"void solve_factored\([^{}]+\) const \{", "solve_factored"),
            ("kkt_product", r"void unregularized_residual\([^{}]+\) const \{", "kkt_product"),
            ("assess_homogeneous", r"bool assess_homogeneous\([^{}]+\) \{", "assess_homogeneous"),
            ("assess", r"bool assess\([^{}]+\) \{", "assess"),
            ("homogeneous_direction", r"HomogeneousDirection homogeneous_direction\([^{}]+\) \{", "homogeneous_direction"),
            ("initialize", r"void initialize\([^{}]+\) \{", "initialize"),
            ("run_homogeneous", r"void run_homogeneous_ipm\([^{}]+\) \{", "run_homogeneous"),
            ("cone_multiply", r"void multiply\([^{}]+\) const \{", "cone_multiply"),
            ("cone_transpose", r"void transpose\([^{}]+\) const \{", "cone_transpose"),
        ):
            scope(label, pattern, stage)
        expression("verify_original", "verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality)", "verify_original")
        # 旧 ordering 仅保留外层 KKT 的 AMD，正规方程的第二次排序另记。
        expression("kkt_ordering", "amd_l_order(static_cast<Index>(factor_dimension_), starts.data(), rows.data(),\n                                      permutation.data(), nullptr, nullptr)", "ordering")
        # 结构回退仍统计适配器，不能把其内部检查时间称作纯数值内核。
        expression("fallback_factor", "multifrontal_ldl_->factor(std::span<const double>(factor_values, ax_.size()),\n            [&options, &start] { check_budget(options, start); })", "fallback_factor_adapter")
        expression("fallback_solve", "multifrontal_ldl_->solve(rhs,\n          [&options, &start] { check_budget(options, start); })", "fallback_solve_adapter")
    else:
        scope("normal_setup", r"Impl\(std::size_t size,[^{}]+max_refinement_steps\(refinement_budget\) \{", "normal_setup")
        scope("normal_factor", r"bool NormalCholesky::factor\([^{}]+\) \{", "normal_factor")
        scope("normal_solve", r"bool NormalCholesky::solve\([^{}]+\) const \{", "normal_solve")
        # 两种内核现在具有同名检查函数；包含各自首行的字面量分别固定边界。
        declaration = "  bool finite_factor(const std::function<void()>& budget_check) const {\n"
        for label, first_line in (
            ("factor_check_eigen", "    // Eigen 的 info() 不保证捕获分解中间产生的 NaN，独立核对存储因子。\n"),
            ("factor_check_cholmod", "    const auto& factor = *resources_.factor;\n"),
        ):
            replace(label, declaration + first_line,
                    declaration + f"    solver_v47_profile::Scope v47_scope({STAGES.index('factor_check')});\n" + first_line)
        expression("normal_ordering", "amd_l_order(retained, matrix.outerIndexPtr(), matrix.innerIndexPtr(), permutation.data(), nullptr, nullptr)", "normal_ordering")
        expression("symbolic_etree", "QDLDL_etree(static_cast<QDLDL_int>(retained), starts.data(), rows.data(), work.data(), counts.data(), tree.data())", "symbolic_analysis")
        expression("symbolic_eigen", "cholesky.analyzePattern(matrix)", "symbolic_analysis")
        expression("symbolic_extended", "extended_cholesky.analyzePattern(extended_matrix)", "symbolic_analysis")
        # 矩形输入先缩放 C，再直接交给 CHOLMOD；它的早返回不得被计作 Gram 装配。
        scaling_pattern = (
            r"      // C=\[G D\^\(-1/2\),sqrt\(R\)\][\s\S]+?"
            r"(?=      check\(budget_check\);\n"
            r"      if \(!state\.cholmod->factor\(std::span<const double>\(state\.rectangular_matrix\.valuePtr\(\),)"
        )
        matches = list(re.finditer(scaling_pattern, source))
        require(len(matches) == 1, "AAt 缩放锚点不唯一")
        body = matches[0].group()
        replace("aat_scaling", body,
                "      {\n        solver_v47_profile::Scope v47_aat_scope(" +
                str(STAGES.index("aat_scaling")) + ");\n" + body + "      }\n")
        pattern = r"  std::fill\(state.sums.begin\(\), state.sums.end\(\), 0\);[\s\S]+?  check\(budget_check\);\n(?=  if \(!state.dual_coordinates.empty\(\)\) \{)"
        matches = list(re.finditer(pattern, source))
        require(len(matches) == 1, "Gram 装配锚点不唯一")
        body = matches[0].group()
        require("state.rectangular" not in body and "state.input_values.assign" not in body,
                "Gram 计时不得包含矩形分支或原输入复制")
        replace("gram_assembly", body,
                "  {\n    solver_v47_profile::Scope v47_gram_scope(" +
                str(STAGES.index("gram_assembly")) + ");\n" + body + "  }\n")
        expression("numeric_double", "state.cholesky.factorize(state.matrix)", "numeric_factor")
        expression("numeric_extended", "state.extended_cholesky.factorize(state.extended_matrix)", "numeric_factor")
        for label, expr, value_type in (
            ("triangular_double", "state.cholesky.solve(permuted)", "Eigen::VectorXd"),
            ("triangular_extended", "state.extended_cholesky.solve(permuted)", "ExtendedVector"),
        ):
            expression(label, expr, "triangular_solve", value_type)
        # 只包同步 C API；调用前后的预算检查、结果校验及恢复留在父作用域。
        # 正常离开计时作用域仍包含库返回失败，不能将 completed 解释为求解成功。
        expression("cholmod_symbolic", "cholmod_l_analyze(&matrix_, &resources_.common)", "cholmod_symbolic")
        numeric_call = "cholmod_l_factorize(input, resources_.factor, &resources_.common)"
        numeric_stage = ("matrix_.stype == 0 ? " + str(STAGES.index("cholmod_aat_numeric")) +
                         " : " + str(STAGES.index("cholmod_numeric")))
        replace("cholmod_numeric", numeric_call,
                "solver_v47_profile::call(" + numeric_stage + ", [&] { return " + numeric_call + "; })")
        expression("cholmod_backsolve", "cholmod_l_solve2(CHOLMOD_A, resources_.factor, &input, nullptr,\n        &resources_.solution, nullptr, &resources_.work, &resources_.extra, &resources_.common)", "cholmod_backsolve")
        copy_body = (
            "      for (std::size_t destination = 0; destination < lower_values_.size(); ++destination) {\n"
            "        if ((destination & 4095) == 0) check(budget_check);\n"
            "        // 仅复制本次已舍入的 double，不缓存跨次分解的矩阵数值。\n"
            "        lower_values_[destination] = values[static_cast<std::size_t>(lower_sources_[destination])];\n"
            "      }\n"
        )
        replace("cholmod_values_copy", copy_body,
                "      {\n        solver_v47_profile::Scope v47_copy_scope(" +
                str(STAGES.index("cholmod_values_copy")) + ");\n" + copy_body + "      }\n")
    rebuilt = source
    for change in reversed(changes):
        rebuilt = replace_once(rebuilt, change["after"], change["before"],
                               f"{unit}/{change['label']} 逆变换")
    require(rebuilt == original, f"{unit} 插桩不可逆")
    return source, changes


def static_check(root):
    """仅验证身份、派生字节和插桩正反例；不编译、不运行 LP，也不写文件。"""
    plan = locked_inputs()
    require(list(STAGES) == plan["profile_stages"], "阶段集合偏离冻结计划")
    restored_header = replace_once(HEADER, _new_capacity, _old_capacity, "头文件容量逆变换")
    restored_header = replace_once(restored_header, _new_names, _old_names, "头文件名称逆变换")
    require(restored_header == _legacy.HEADER, "计时核心不得改变")
    require(SELF_MAIN == _legacy.SELF_MAIN and SELF_CROSS == _legacy.SELF_CROSS,
            "双翻译单元自测不得改变")
    records = {}
    for unit, relative in plan["transformed_units"].items():
        original = (Path(root) / relative).read_text(encoding="utf-8")
        patched, changes = transform(original, unit)
        records[unit] = {"anchors": len(changes), "source_text_sha256": digest(original),
                         "instrumented_sha256": digest(patched), "reversible": True}
        for modified in (original.replace("namespace solver", "namespace removed", 1),
                         original + original, patched):
            rejected = False
            try:
                transform(modified, unit)
            except RuntimeError:
                rejected = True
            require(rejected, "必须拒绝缺失、重复锚点及已插桩的输入")
    return {"passed": True, "units": records, "stages": list(STAGES),
            "header_sha256": digest(HEADER), "legacy_helper_sha256": HELPER_SHA256,
            "plan_sha256": PLAN_SHA256, "header_core_unchanged": True,
            "self_main_sha256": digest(SELF_MAIN), "self_cross_sha256": digest(SELF_CROSS),
            "scope": "仅静态身份和变换正反例；未编译、未运行 LP"}


if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False))
