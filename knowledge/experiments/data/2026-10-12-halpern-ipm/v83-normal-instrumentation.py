"""为冻结正规方程实现添加双右端接口；不改原单右端表达式和数值门槛。"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "46e074b8cab1b8f02772cb00b737d9c91d49dd7f90ddc14c99d87a913922308b"
UNITS = {"normal": "src/normal_cholesky.cpp",
         "normal_header": "include/solver/detail/normal_cholesky.hpp"}

HEADER_SINGLE = "  bool solve(std::vector<double>& rhs, const std::function<void()>& budget_check = {}) const;\n"
HEADER_PAIR = """  // 能力查询针对实际内核；未支持时双右端接口立即返回 false，不启动工作。
  bool supports_native_pair() const;
  // 仅完整双回代结果均有限且可表示时同时写回；异常保留原类型。
  bool solve_pair(std::array<std::vector<double>, 2>& rhs,
                  const std::function<void()>& budget_check = {}) const;
"""

CHOLMOD_PAIR = """  bool solve_pair(std::array<std::vector<double>, 2>& rhs,
                  const std::function<void()>& budget_check) const {
    if (!valid_) return false;
    check(budget_check);
    const auto retained = matrix_.nrow;
    if (rhs[0].size() != retained || rhs[1].size() != retained ||
        !std::all_of(rhs[0].begin(), rhs[0].end(), [](double value) { return std::isfinite(value); }) ||
        !std::all_of(rhs[1].begin(), rhs[1].end(), [](double value) { return std::isfinite(value); })) return false;
    std::vector<double> packed;
    if (retained > packed.max_size() / 2) throw std::length_error("CHOLMOD 双右端尺寸无法表示");
    packed.resize(2 * retained);
    for (std::size_t row = 0; row < retained; ++row) {
      if ((row & 4095) == 0) check(budget_check);
      packed[row] = rhs[0][row];
      packed[retained + row] = rhs[1][row];
    }
    cholmod_dense input{};
    input.nrow = input.d = retained;
    input.ncol = 2;
    input.nzmax = packed.size();
    input.x = packed.data();
    input.xtype = CHOLMOD_REAL;
    input.dtype = CHOLMOD_DOUBLE;
    check(budget_check);
    const int success = cholmod_l_solve2(CHOLMOD_A, resources_.factor, &input, nullptr,
        &resources_.solution, nullptr, &resources_.work, &resources_.extra, &resources_.common);
    check(budget_check);
    check_status();
    const auto* output = resources_.solution;
    if (!success || resources_.common.status != CHOLMOD_OK || !output || !output->x ||
        output->nrow != retained || output->ncol != 2 || output->d < retained ||
        output->d > std::numeric_limits<std::size_t>::max() / 2 || output->nzmax < 2 * output->d ||
        output->xtype != CHOLMOD_REAL || output->dtype != CHOLMOD_DOUBLE) return false;
    const auto* values = static_cast<const double*>(output->x);
    std::array<std::vector<double>, 2> solution{std::vector<double>(retained), std::vector<double>(retained)};
    for (std::size_t row = 0; row < retained; ++row) {
      if ((row & 4095) == 0) check(budget_check);
      const double first = values[row], second = values[output->d + row];
      if (!std::isfinite(first) || !std::isfinite(second)) return false;
      solution[0][row] = first;
      solution[1][row] = second;
    }
    // 两列都独立拥有内存，之后单右端精修复用库缓存不会使第二列悬空。
    check(budget_check);
    rhs.swap(solution);
    return true;
  }

"""

NORMAL_PAIR = """bool NormalCholesky::supports_native_pair() const {
  const auto& state = *impl_;
  return state.cholmod != nullptr && !state.extended && state.max_refinement_steps == 0 &&
      state.multifrontal == nullptr;
}

bool NormalCholesky::solve_pair(std::array<std::vector<double>, 2>& rhs,
                                const std::function<void()>& budget_check) const {
  // 未支持的路径不执行回调、不修改 RHS，也不启动 Schur 校正统计。
  if (!supports_native_pair()) return false;
  const auto& state = *impl_;
  check(budget_check);
  if (!state.valid || rhs[0].size() != state.dimension || rhs[1].size() != state.dimension ||
      !std::all_of(rhs[0].begin(), rhs[0].end(), [](double value) { return std::isfinite(value); }) ||
      !std::all_of(rhs[1].begin(), rhs[1].end(), [](double value) { return std::isfinite(value); })) return false;
  const auto retained = state.dual_coordinates.size();
  std::array<std::vector<long double>, 2> reduced_rhs{
      std::vector<long double>(retained), std::vector<long double>(retained)};
  for (std::size_t row = 0; row < retained; ++row) {
    if ((row & 4095) == 0) check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.dual_coordinates[row]);
    reduced_rhs[0][row] = -static_cast<long double>(rhs[0][coordinate]);
    reduced_rhs[1][row] = -static_cast<long double>(rhs[1][coordinate]);
  }
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
    const auto diagonal = static_cast<std::size_t>(state.primal_diagonal[col]);
    const long double first = static_cast<long double>(rhs[0][coordinate]) / state.input_values[diagonal];
    const long double second = static_cast<long double>(rhs[1][coordinate]) / state.input_values[diagonal];
    if (!std::isfinite(first) || !std::isfinite(second)) return false;
    std::size_t cursor = 0;
    for (const auto& entry : state.columns[col]) {
      if (cursor != 0 && (cursor & 255) == 0) check(budget_check);
      ++cursor;
      reduced_rhs[0][static_cast<std::size_t>(entry.row)] += state.input_values[static_cast<std::size_t>(entry.source)] * first;
      reduced_rhs[1][static_cast<std::size_t>(entry.row)] += state.input_values[static_cast<std::size_t>(entry.source)] * second;
    }
  }
  std::array<std::vector<double>, 2> permuted{std::vector<double>(retained), std::vector<double>(retained)};
  for (std::size_t row = 0; row < retained; ++row) {
    if ((row & 4095) == 0) check(budget_check);
    const auto position = static_cast<std::size_t>(state.inverse[row]);
    if (!representable(reduced_rhs[0][row], permuted[0][position]) ||
        !representable(reduced_rhs[1][row], permuted[1][position])) return false;
  }
  check(budget_check);
  if (!state.cholmod->solve_pair(permuted, budget_check)) return false;
  std::array<std::vector<double>, 2> solution{
      std::vector<double>(rhs[0].size()), std::vector<double>(rhs[1].size())};
  for (std::size_t row = 0; row < retained; ++row) {
    if ((row & 4095) == 0) check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.dual_coordinates[row]);
    const auto position = static_cast<std::size_t>(state.inverse[row]);
    solution[0][coordinate] = permuted[0][position];
    solution[1][coordinate] = permuted[1][position];
  }
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
    long double first = rhs[0][coordinate], second = rhs[1][coordinate];
    std::size_t cursor = 0;
    for (const auto& entry : state.columns[col]) {
      if (cursor != 0 && (cursor & 255) == 0) check(budget_check);
      ++cursor;
      const auto position = static_cast<std::size_t>(state.inverse[static_cast<std::size_t>(entry.row)]);
      first -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) * permuted[0][position];
      second -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) * permuted[1][position];
    }
    first /= state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    second /= state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    if (!representable(first, solution[0][coordinate]) || !representable(second, solution[1][coordinate])) return false;
  }
  // 两列的原恢复与表示检查全部通过后，常数时间交换同时提交。
  check(budget_check);
  rhs.swap(solution);
  return true;
}

"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(source, unit):
    """所有新增接口均有唯一锚点；逆变换恢复规范换行后的整个冻结文件。"""
    require(unit in UNITS, "未知的正规方程翻译单元")
    source = source.replace("\r\n", "\n")
    require("supports_native_pair" not in source and "solve_pair" not in source, "拒绝重复双右端插桩")
    original, changes = source, []

    def replace(label, before, after):
        nonlocal source
        require(source.count(before) == 1 and before != after, "双右端锚点缺失或不唯一：" + label)
        source = source.replace(before, after, 1)
        changes.append({"label": label, "before": before, "after": after})

    if unit == "normal_header":
        replace("array_include", "#include <cstddef>\n", "#include <array>\n#include <cstddef>\n")
        replace("pair_declarations", HEADER_SINGLE, HEADER_SINGLE + HEADER_PAIR)
    else:
        anchor = "  Index stored_nonzeros() const { return static_cast<Index>(resources_.factor->xsize); }"
        replace("cholmod_pair", anchor, CHOLMOD_PAIR + anchor)
        anchor = "Index NormalCholesky::factor_nonzeros() const"
        replace("normal_pair", anchor, NORMAL_PAIR + anchor)
    restored = source
    for change in reversed(changes):
        require(restored.count(change["after"]) == 1, "双右端逆变换锚点不唯一")
        restored = restored.replace(change["after"], change["before"], 1)
    require(restored == original, "双右端逆变换未恢复完整单右端源码")
    return source, changes


def static_check(root):
    """只读验证来源和可逆接口变换，不编译、不执行测试或 LP，不写文件。"""
    records = {}
    for unit, relative in UNITS.items():
        original = (Path(root) / relative).read_text(encoding="utf-8")
        patched, changes = transform(original, unit)
        require(len(changes) == 2, "每文件恰两个接口插入")
        if unit == "normal_header":
            require("#include <array>" in patched and HEADER_SINGLE in patched and HEADER_PAIR in patched,
                    "头文件只追加容器依赖和两个声明")
        else:
            require(patched.count("input.ncol = 2;") == 1 and patched.count("rhs.swap(solution);") == 2,
                    "仅一个原生双右端库调用与两层事务提交")
            require("if (!supports_native_pair()) return false;" in patched and
                    "state.cholmod->solve_pair(permuted, budget_check)" in patched,
                    "未支持路径立即返回，原生调用只走当前 CHOLMOD")
        records[unit] = {"source_text_sha256": digest(original), "instrumented_sha256": digest(patched),
                         "anchors": len(changes), "reversible": True}
        for invalid in (original + original, patched, original.replace(changes[0]["before"], "", 1)):
            rejected = False
            try:
                transform(invalid, unit)
            except RuntimeError:
                rejected = True
            require(rejected, "必须拒绝重复、已插桩和缺失锚点")
    return {"passed": True, "public_lp_runs": 0, "units": records, "plan_sha256": PLAN_SHA256,
            "single_rhs_source_unchanged": True, "native_nrhs": 2,
            "unsupported_no_work": True, "dual_rhs_commit": True,
            "scope": "仅静态可逆接口检查；未编译、未运行数值或 LP"}


if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False))
